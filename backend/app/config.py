import os

DEFAULT_JWT_SECRET = "dev-secret-change-me"
DEFAULT_ADMIN_PASSWORD = "admin12345"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    """All configuration comes from environment variables (12-factor style)."""

    def __init__(self) -> None:
        e = os.getenv
        self.env = e("APP_ENV", "dev")
        self.database_url = e("DATABASE_URL", "sqlite:///./app.db")

        self.jwt_secret = e("JWT_SECRET", DEFAULT_JWT_SECRET)
        self.jwt_algorithm = "HS256"
        self.access_token_minutes = int(e("ACCESS_TOKEN_MINUTES", "30"))
        self.refresh_token_days = int(e("REFRESH_TOKEN_DAYS", "7"))

        self.cors_origins = _csv(e("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"))

        self.admin_username = e("ADMIN_USERNAME", "admin")
        self.admin_password = e("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)

        self.redis_url = e("REDIS_URL", "redis://localhost:6379/0")
        self.celery_queue = e("CELERY_QUEUE", "celery")
        self.celery_task_name = e("CELERY_TASK_NAME", "worker.process_job")

        self.max_upload_mb = int(e("MAX_UPLOAD_MB", "10"))

        self.storage_backend = e("STORAGE_BACKEND", "local")
        self.local_storage_dir = e("LOCAL_STORAGE_DIR", "./storage")
        self.r2_endpoint = e("R2_ENDPOINT", "")
        self.r2_access_key = e("R2_ACCESS_KEY", "")
        self.r2_secret_key = e("R2_SECRET_KEY", "")
        self.r2_bucket = e("R2_BUCKET", "")

        self.hf_webhook_secret = e("HF_WEBHOOK_SECRET", "")
        self.pipeline_token = e("PIPELINE_TOKEN", "")
        self.min_map_improvement = float(e("MIN_MAP_IMPROVEMENT", "0.0"))

        self.github_repo = e("GITHUB_REPO", "")
        self.github_token = e("GITHUB_TOKEN", "")
        self.github_workflow = e("GITHUB_WORKFLOW", "train.yml")
        self.github_ref = e("GITHUB_REF", "main")

        if self.env == "prod":
            if self.jwt_secret == DEFAULT_JWT_SECRET:
                raise RuntimeError("JWT_SECRET must be set in production")
            if self.admin_password == DEFAULT_ADMIN_PASSWORD:
                raise RuntimeError("ADMIN_PASSWORD must be set in production")


settings = Settings()
