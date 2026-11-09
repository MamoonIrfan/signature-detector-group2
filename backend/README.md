# Signature Detection and Verification API (Group 2)

FastAPI backend with JWT auth, CORS, async job queue (Celery + Redis), job table, HF webhook,
training trigger and a model registry with an evaluation gate.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit secrets
uvicorn app.main:app --reload   # http://localhost:8000/docs
pytest -q
```

Or with Docker: `cp .env.example .env && docker compose up --build`

Default admin (dev only): username/password from `ADMIN_USERNAME` / `ADMIN_PASSWORD`.
With `APP_ENV=prod` the app refuses to start on default secrets.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | /health | none | liveness + DB check |
| POST | /auth/login | none | JSON `{username, password}` -> access + refresh tokens |
| POST | /auth/refresh | refresh token | new token pair |
| GET | /auth/me | access | current user |
| POST | /auth/register | admin | create user |
| POST | /detect | access | upload `file` (PNG/JPEG) -> `202 {job_id, status}` |
| POST | /compare | access | upload `reference` + `test` -> `202 {job_id, status}` |
| GET | /jobs/{id} | access | status + result (owner or admin only) |
| GET | /jobs | access | recent jobs (`?status=`, `?limit=`) |
| POST | /webhooks/hf | `X-Webhook-Secret` | HF dataset update -> trigger training (deduplicated by commit sha) |
| POST | /training/trigger | admin | manual backup trigger |
| GET | /training/runs | admin | training run history |
| POST | /models/promote | `X-Pipeline-Token` or admin | evaluation gate (map50 must beat current) |
| GET | /models/current | access | current model version and URL |
| GET | /models | access | all versions |
| POST | /models/rollback | `X-Pipeline-Token` or admin | restore previous model |

`POST /detect` and `/compare` accept an optional `Idempotency-Key` header: the same key returns the same job.

## Contract for the worker (Farhan)

- Task name: `worker.process_job`, args `[job_id]`, queue from `CELERY_QUEUE` (override with `CELERY_TASK_NAME`).
- The job row already exists (committed before enqueue). Read `jobs.input_keys` (storage keys, e.g.
  `{"image": "uploads/<job_id>/image.png"}` or `{"reference": ..., "test": ...}`) and load bytes with `app.storage.get_storage().load(key)`.
- Update the same row: `status` = `processing` -> `done` or `failed`; on success set `result_json`, on failure set `error`.
- Idempotency: if `status` is already `done` or `processing`, skip.
- Suggested `result_json`:
  - detect: `{"detections": [{"bbox": [x1,y1,x2,y2], "score": 0.93, "crop_key": "crops/<job_id>/0.png"}]}`
  - compare: `{"similarity": 0.87, "match": true, "threshold": 0.8}`
- Hot reload: read the current model from `model_versions` where `is_current = true` (or `GET /models/current`).

## Contract for the training pipeline (Zubair)

After training and evaluation, call:

```bash
curl -X POST $API/models/promote -H "X-Pipeline-Token: $PIPELINE_TOKEN" -H "Content-Type: application/json" \
  -d '{"version":"2026-11-09-a1b2c3","url":"s3://models/yolox_s_a1b2c3.onnx","metrics":{"map50":0.91},"run_id":"mlflow-run-id"}'
```
The response says whether the model was promoted. The workflow receives `dataset_repo` and `dataset_sha` as inputs.

## Hugging Face webhook setup

1. HF settings -> Webhooks -> new webhook, watch the dataset repo, event "Repo content updates".
2. URL: `https://<your-api>/webhooks/hf`, secret = `HF_WEBHOOK_SECRET`.
3. Set `GITHUB_REPO`, `GITHUB_TOKEN`, `GITHUB_WORKFLOW` so the API can dispatch the training workflow.
   Without them runs are recorded as `pending_manual`.
4. Local testing: `ngrok http 8000`, then use the ngrok URL.

## Design notes (for the teacher)

- Job is committed to the DB before it is queued, so the worker never sees a missing row. If the broker is down the job is marked `failed` and the API returns 503.
- Uploads are validated by file signature (PNG/JPEG) and size, not by client content-type.
- Access and refresh tokens carry a `type` claim, so a refresh token cannot be used as an access token.
- Jobs of other users return 404, so job ids are not revealed.
- Webhook is deduplicated by (repo, commit sha), so HF retries do not start training twice.

## Dependency licenses

| Package | License |
|---|---|
| FastAPI, SQLAlchemy, PyJWT, redis-py, python-multipart | MIT |
| Uvicorn, httpx, Celery | BSD-3-Clause |
| bcrypt, boto3 | Apache-2.0 |
| psycopg2-binary | LGPL-3.0 (dynamic linking is allowed; note it in the audit) |

Verify licenses again from each package page before submission.
