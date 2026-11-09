import hmac
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import User

bearer = HTTPBearer(auto_error=False)


def _pw(password: str) -> bytes:
    # bcrypt only uses the first 72 bytes
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_pw(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_pw(password), password_hash.encode("utf-8"))
    except ValueError:
        return False


# Used so that unknown usernames take the same time as wrong passwords
DUMMY_HASH = hash_password("dummy-password-for-timing")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_token(sub: str, role: str, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "role": role, "type": token_type, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token expired")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid token")
    if payload.get("type") != expected_type:
        raise _unauthorized("Wrong token type")
    return payload


def issue_token_pair(user: User) -> tuple[str, str]:
    access = create_token(user.username, user.role, "access", timedelta(minutes=settings.access_token_minutes))
    refresh = create_token(user.username, user.role, "refresh", timedelta(days=settings.refresh_token_days))
    return access, refresh


def _user_from_credentials(creds: HTTPAuthorizationCredentials | None, db: Session) -> User:
    if creds is None:
        raise _unauthorized("Not authenticated")
    payload = decode_token(creds.credentials, "access")
    user = db.scalar(select(User).where(User.username == payload.get("sub")))
    if user is None:
        raise _unauthorized("User no longer exists")
    return user


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    return _user_from_credentials(creds, db)


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def require_pipeline_or_admin(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> str:
    """Training pipeline authenticates with X-Pipeline-Token; humans with an admin JWT."""
    token = request.headers.get("X-Pipeline-Token")
    if token and settings.pipeline_token and hmac.compare_digest(token, settings.pipeline_token):
        return "pipeline"
    user = _user_from_credentials(creds, db)
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin or pipeline token required")
    return user.username
