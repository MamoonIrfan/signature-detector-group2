import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # worker ko .env ki values chahiye

import redis  # noqa: E402
from celery import Task  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.job_queue import celery_app  # noqa: E402
from app.models import DONE, FAILED, JOB_TYPE_DETECT, PROCESSING, Job  # noqa: E402
from app.storage import get_storage  # noqa: E402

app = celery_app
# Crash ke baad unacked job itni der me wapas milti hai
celery_app.conf.broker_transport_options = {"visibility_timeout": 60}

# --- JSON logging ---
log = logging.getLogger("worker")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def log_event(job_id, status, **extra):
    log.info(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id": job_id,
        "status": status,
        **extra,
    }))


# --- failed_jobs list (DLQ jaisi) ---
_base = settings.redis_url.split("?")[0]
_redis = redis.from_url(
    _base,
    decode_responses=True,
    **({"ssl_cert_reqs": "required"} if _base.startswith("rediss") else {}),
)


def run_detect(job_id, image_key):
    from detect import detect  # Zubair ka detect.py

    storage = get_storage()
    detections = []
    for i, d in enumerate(detect(storage.load(image_key))):
        crop_path = Path(d["crop_path"])
        crop_key = f"crops/{job_id}/signature_{i:03d}.png"
        storage.save(crop_key, crop_path.read_bytes(), "image/png")
        crop_path.unlink(missing_ok=True)
        detections.append({"bbox": d["bbox"], "score": d["score"], "crop_key": crop_key})
    return {"detections": detections}


class JobTask(Task):
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        log_event(args[0], "retrying", error=str(exc), attempt=self.request.retries + 1)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0]
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is not None:
                job.status = FAILED
                job.error = str(exc)[:500]
                db.commit()
        _redis.lpush("failed_jobs", json.dumps(
            {"job_id": job_id, "error": str(exc), "ts": time.time()}))
        log_event(job_id, "failed", error=str(exc))


@celery_app.task(name=settings.celery_task_name, bind=True, base=JobTask, acks_late=True,
                 autoretry_for=(Exception,), retry_backoff=True, retry_backoff_max=60,
                 retry_jitter=False, max_retries=3)
def process_job(self, job_id):
    redelivered = bool((self.request.delivery_info or {}).get("redelivered"))
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            log_event(job_id, "skipped", reason="job not found")
            return
        if job.status in (DONE, FAILED):
            log_event(job_id, "skipped_duplicate", reason=f"already {job.status}")
            return
        # Retry ya crash ke baad redeliver hui job chalne do, nayi duplicate nahi
        if job.status == PROCESSING and self.request.retries == 0 and not redelivered:
            log_event(job_id, "skipped_duplicate", reason="already processing")
            return
        if redelivered:
            log_event(job_id, "redelivered", note="picked up after worker crash")

        job.status = PROCESSING
        db.commit()
        log_event(job_id, "processing", attempt=self.request.retries + 1)

        start = time.time()
        if job.job_type == JOB_TYPE_DETECT:
            result = run_detect(job_id, job.input_keys["image"])
        else:
            # compare(): Umme ka function aane tak dummy
            result = {"similarity": None, "note": "compare() abhi ready nahi"}

        job.result_json = result
        job.status = DONE
        job.error = None
        db.commit()
        log_event(job_id, "done", duration=round(time.time() - start, 2))