import time
from celery_app import celery_app

@celery_app.task(bind=True)
def detect_signature(self, image_path: str):
    time.sleep(5)  # placeholder, baad me asli detection model yahan aayega
    return {"image": image_path, "signatures_found": 1, "boxes": [[10, 10, 100, 50]]}

STATE_MAP = {"PENDING": "queued", "STARTED": "processing",
             "SUCCESS": "done", "FAILURE": "failed"}

def get_job_status(job_id: str):
    r = celery_app.AsyncResult(job_id)
    out = {"job_id": job_id, "status": STATE_MAP.get(r.state, r.state.lower())}
    if r.state == "SUCCESS":
        out["result"] = r.result
    if r.state == "FAILURE":
        out["error"] = str(r.result)
    return out