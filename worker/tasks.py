import os, json, time, uuid, logging
import redis
from celery import Task
from dotenv import load_dotenv
from celery_app import celery_app

load_dotenv()

# --- API database (jobs table) --- [added for API integration]
# Imported after load_dotenv() so DATABASE_URL etc. are already in the environment.
from app.db import SessionLocal  # noqa: E402
from app.models import Job  # noqa: E402

# --- JSON logging ---
log = logging.getLogger("worker")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# --- Redis ---
_url = os.getenv("REDIS_URL", "")
_base = _url.split("?")[0]
r = redis.from_url(
    _base,
    decode_responses=True,
    **({"ssl_cert_reqs": "required"} if _base.startswith("rediss") else {}),
)


def log_event(job_id, status, **extra):
    log.info(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id": job_id,
        "status": status,
        **extra,
    }))


# --- [added for API integration] mirror job state into the API's `jobs` table ---
def sync_db(job_id, status, **extra):
    """Keeps GET /jobs/{id} in sync. Does nothing if the job has no DB row (e.g. test_flow.py)."""
    db_status = "processing" if status == "retrying" else status
    try:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.status = db_status
            if status == "done":
                result = extra.get("result") or {}
                if "detections" not in result:  # dummy format -> API contract format
                    result = {**result, "detections": [{"bbox": b, "score": None} for b in result.get("boxes", [])]}
                job.result_json = result
                job.error = None
            elif status in ("failed", "retrying"):
                job.error = extra.get("error")
            db.commit()
    except Exception:
        log.exception("db sync failed job_id=%s status=%s", job_id, status)
        if status == "done":
            raise  # let Celery retry, otherwise the API would show 'processing' forever


def load_input_key(job_id):
    """[added] The API sends only job_id; the input image lives in storage, its key is in the jobs table."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found in database")
        if job.job_type != "detect":
            raise NotImplementedError(f"job type '{job.job_type}' is not supported yet")
        return job.input_keys["image"]


def set_status(job_id, status, **extra):
    sync_db(job_id, status, **extra)          # [added] DB first, then Redis
    r.set(f"job:{job_id}",
          json.dumps({"job_id": job_id, "status": status, **extra}),
          ex=86400)
    log_event(job_id, status, **extra)


def get_status(job_id):
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


def detect(image_path):
    # Dummy. Zubair ka asli detect() aane par yahan replace hoga.
    name = os.path.basename(image_path)   # [changed] storage keys contain a random uuid, which could contain "bad"
    if "bad" in name:
        raise ValueError("corrupt image")      # retry test ke liye
    time.sleep(15 if "slow" in name else 3)   # slow = crash test ke liye
    return {"image": image_path, "signatures_found": 1, "boxes": [[10, 10, 100, 50]]}


class JobTask(Task):
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        set_status(args[0], "retrying", error=str(exc),
                   attempt=self.request.retries + 1)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0]
        set_status(job_id, "failed", error=str(exc))
        r.lpush("failed_jobs", json.dumps(
            {"job_id": job_id, "args": list(args),
             "error": str(exc), "ts": time.time()}))
        r.delete(f"lock:{job_id}")


# [changed] explicit task name (the API sends "worker.process_job"); image_path is now optional
@celery_app.task(name="worker.process_job", bind=True, base=JobTask, acks_late=True,
                 autoretry_for=(Exception,), retry_backoff=True,
                 retry_backoff_max=60, retry_jitter=False, max_retries=3)
def process_job(self, job_id, image_path=None):
    state = get_status(job_id)
    if state and state["status"] == "done":
        log_event(job_id, "skipped_duplicate", reason="already done")
        return state.get("result")

    if image_path is None:                      # [added] job came from the API
        image_path = load_input_key(job_id)

    # Idempotency: pehli attempt par lock. Crash ke baad redeliver hui job ko lock se nahi roka jata.
    redelivered = bool((self.request.delivery_info or {}).get("redelivered"))
    if self.request.retries == 0 and not redelivered:
        if not r.set(f"lock:{job_id}", "1", nx=True, ex=60):
            log_event(job_id, "skipped_duplicate", reason="already processing")
            return None
    else:
        r.set(f"lock:{job_id}", "1", ex=60)
        if redelivered:
            log_event(job_id, "redelivered", note="picked up after worker crash")

    set_status(job_id, "processing", attempt=self.request.retries + 1)
    start = time.time()
    result = detect(image_path)
    set_status(job_id, "done", result=result,
               duration=round(time.time() - start, 2))
    r.delete(f"lock:{job_id}")
    return result


def submit_job(image_path, job_id=None):
    job_id = job_id or str(uuid.uuid4())
    existing = get_status(job_id)
    if existing and existing["status"] in ("queued", "processing", "retrying", "done"):
        log_event(job_id, "skipped_duplicate", reason=f"already {existing['status']}")
        return job_id
    set_status(job_id, "queued")
    process_job.apply_async(args=[job_id, image_path], task_id=job_id)
    return job_id


def get_job_status(job_id):
    return get_status(job_id) or {"job_id": job_id, "status": "not_found"}
