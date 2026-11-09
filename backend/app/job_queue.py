"""Publishes jobs to the broker. The worker (Farhan) consumes `worker.process_job`."""
from celery import Celery

from app.config import settings

celery_app = Celery("api", broker=settings.redis_url)
celery_app.conf.broker_connection_timeout = 3


def enqueue_job(job_id: str) -> None:
    """Send the job to the worker. Raises if the broker is unreachable."""
    celery_app.send_task(
        settings.celery_task_name,
        args=[job_id],
        queue=settings.celery_queue,
        retry=False,
    )
