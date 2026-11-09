from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: Literal["user", "admin"] = "user"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    username: str
    role: str


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    job_id: str
    job_type: str
    status: str
    result_json: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class TrainTriggerRequest(BaseModel):
    dataset_repo: str = Field(min_length=1, max_length=256)
    dataset_sha: str | None = Field(default=None, max_length=128)


class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    dataset_repo: str
    dataset_sha: str
    status: str
    detail: str | None = None
    created_at: datetime


class PromoteRequest(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=512)
    metrics: dict[str, float]
    run_id: str | None = Field(default=None, max_length=128)


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    version: str
    url: str
    metrics: dict[str, Any]
    run_id: str | None = None
    is_current: bool
    created_at: datetime
    promoted_at: datetime | None = None


class PromoteResponse(BaseModel):
    promoted: bool
    reason: str
    model: ModelOut
