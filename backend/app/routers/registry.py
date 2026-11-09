import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_user, require_pipeline_or_admin
from app.config import settings
from app.db import get_db
from app.models import ModelVersion, User, utcnow
from app.schemas import ModelOut, PromoteRequest, PromoteResponse

logger = logging.getLogger("api.registry")
router = APIRouter(prefix="/models", tags=["model registry"])


def _current(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.is_current.is_(True)))


@router.post("/promote", response_model=PromoteResponse)
def promote(body: PromoteRequest, db: Session = Depends(get_db), _: str = Depends(require_pipeline_or_admin)):
    """Evaluation gate: the new model becomes current only if its mAP@0.5 beats the current model."""
    if "map50" not in body.metrics:
        raise HTTPException(422, "metrics must include 'map50'")
    if db.scalar(select(ModelVersion).where(ModelVersion.version == body.version)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Version already registered")

    current = _current(db)
    new_map = body.metrics["map50"]
    if current is None:
        promote_it, reason = True, "first model, promoted"
    else:
        current_map = float(current.metrics.get("map50", 0.0))
        needed = current_map + settings.min_map_improvement
        if new_map > needed:
            promote_it, reason = True, f"promoted: map50 {new_map:.4f} > current {current_map:.4f}"
        else:
            promote_it, reason = False, f"rejected: map50 {new_map:.4f} does not beat current {current_map:.4f}"

    model = ModelVersion(
        version=body.version,
        url=body.url,
        metrics=body.metrics,
        run_id=body.run_id,
        is_current=promote_it,
        promoted_at=utcnow() if promote_it else None,
    )
    if promote_it and current:
        current.is_current = False
    db.add(model)
    db.commit()
    logger.info("model %s: %s", body.version, reason)
    return PromoteResponse(promoted=promote_it, reason=reason, model=model)


@router.get("/current", response_model=ModelOut)
def current_model(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    model = _current(db)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No model has been promoted yet")
    return model


@router.get("", response_model=list[ModelOut])
def list_models(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())))


@router.post("/rollback", response_model=ModelOut)
def rollback(db: Session = Depends(get_db), _: str = Depends(require_pipeline_or_admin)):
    """Make the previously promoted model current again."""
    current = _current(db)
    if current is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No current model")
    previous = db.scalar(
        select(ModelVersion)
        .where(ModelVersion.promoted_at.is_not(None), ModelVersion.id != current.id)
        .order_by(ModelVersion.promoted_at.desc())
    )
    if previous is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "No previous model to roll back to")
    current.is_current = False
    previous.is_current = True
    db.commit()
    logger.info("rolled back %s -> %s", current.version, previous.version)
    return previous
