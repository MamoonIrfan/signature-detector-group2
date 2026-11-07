import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL, include=["tasks"])
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)