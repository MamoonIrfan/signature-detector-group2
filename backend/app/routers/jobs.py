import logging
import uuid

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import settings
from app.db import get_db
from app.job_queue import enqueue_job
from app.models import FAILED, JOB_TYPE_COMPARE, JOB_TYPE_DETECT, QUEUED, Job, User
from app.schemas import JobCreated, JobOut
from app.storage import get_storage

logger = logging.getLogger("api.jobs")
router = APIRouter(tags=["jobs"])

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


async def _read_image(upload: UploadFile) -> tuple[bytes, str, str]:
    """Read and validate an uploaded image. Validates real file bytes, not the client's content-type."""
    limit = settings.max_upload_mb * 1024 * 1024
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    if data.startswith(PNG_MAGIC):
        return data, "png", "image/png"
    if data.startswith(JPEG_MAGIC):
        return data, "jpg", "image/jpeg"
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PNG and JPEG images are supported")


def _create_job(
    db: Session,
    user: User,
    job_type: str,
    files: dict[str, tuple[bytes, str, str]],
    idempotency_key: str | None,
) -> Job:
    if idempotency_key:
        existing = db.scalar(select(Job).where(Job.owner_id == user.id, Job.idempotency_key == idempotency_key))
        if existing:
            return existing

    job_id = str(uuid.uuid4())
    storage = get_storage()
    keys: dict[str, str] = {}
    for name, (data, ext, content_type) in files.items():
        key = f"uploads/{job_id}/{name}.{ext}"
        storage.save(key, data, content_type)
        keys[name] = key

    job = Job(
        job_id=job_id,
        job_type=job_type,
        status=QUEUED,
        owner_id=user.id,
        input_keys=keys,
        idempotency_key=idempotency_key,
    )
    db.add(job)
    try:
        db.commit()  # commit BEFORE enqueue so the worker can always find the row
    except IntegrityError:
        db.rollback()
        existing = db.scalar(select(Job).where(Job.owner_id == user.id, Job.idempotency_key == idempotency_key))
        if existing:
            return existing
        raise

    try:
        enqueue_job(job_id)
    except Exception:
        logger.exception("enqueue failed", extra={"job_id": job_id})
        job.status = FAILED
        job.error = "Queue unavailable, please retry"
        db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Queue unavailable, please retry")

    logger.info("job queued job_id=%s type=%s owner=%s", job_id, job_type, user.username)
    return job


@router.post("/detect", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def detect(
    file: UploadFile = File(..., description="Document image (PNG or JPEG)"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    image = await _read_image(file)
    job = _create_job(db, user, JOB_TYPE_DETECT, {"image": image}, idempotency_key)
    return JobCreated(job_id=job.job_id, status=job.status)


@router.post("/compare", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def compare(
    reference: UploadFile = File(..., description="Reference signature image"),
    test: UploadFile = File(..., description="Signature image to verify"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ref_image = await _read_image(reference)
    test_image = await _read_image(test)
    job = _create_job(db, user, JOB_TYPE_COMPARE, {"reference": ref_image, "test": test_image}, idempotency_key)
    return JobCreated(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(Job, job_id)
    # 404 (not 403) so job ids of other users are not revealed
    if job is None or (user.role != "admin" and job.owner_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if user.role != "admin":
        query = query.where(Job.owner_id == user.id)
    if status_filter:
        query = query.where(Job.status == status_filter)
    return list(db.scalars(query))
