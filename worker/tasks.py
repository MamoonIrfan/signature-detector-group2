import os, json, time, uuid, logging
import redis
from celery import Task
from dotenv import load_dotenv
from celery_app import celery_app

load_dotenv()
log = logging.getLogger("worker")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

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


def set_status(job_id, status, **extra):
    r.set(f"job:{job_id}",
          json.dumps({"job_id": job_id, "status": status, **extra}),
          ex=86400)
    log_event(job_id, status, **extra)


def get_status(job_id):
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


def detect(image_path):
    # Dummy. Zubair ka asli detect() aane par yahan replace hoga.
    if "bad" in image_path:
        raise ValueError("corrupt image")   # retry test ke liye
    time.sleep(3)
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


@celery_app.task(bind=True, base=JobTask, acks_late=True,
                 autoretry_for=(Exception,), retry_backoff=True,
                 retry_backoff_max=60, retry_jitter=True, max_retries=3)
def process_job(self, job_id, image_path):
    state = get_status(job_id)
    if state and state["status"] == "done":
        log_event(job_id, "skipped_duplicate", reason="already done")
        return state.get("result")
    # Idempotency: pehli attempt par lock lo, retries usi lock ke saath chalti hain
    if self.request.retries == 0 and not r.set(f"lock:{job_id}", "1", nx=True, ex=600):
        log_event(job_id, "skipped_duplicate", reason="already processing")
        return None
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