from dotenv import load_dotenv

load_dotenv()  # worker ko .env ki values chahiye (Celery uvicorn ki --env-file nahi parhta)

from app.db import SessionLocal  # noqa: E402
from app.job_queue import celery_app  # noqa: E402
from app.models import Job  # noqa: E402

app = celery_app


@celery_app.task(name="worker.process_job")
def process_job(job_id):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None or job.status in ("done", "processing"):
            return  # idempotency: dobara process nahi karna
        job.status = "processing"
        db.commit()

        # Yahan baad mein asli detect() / compare() lagega
        job.result_json = {"detections": [], "note": "dummy result"}
        job.status = "done"
        db.commit()