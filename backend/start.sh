#!/bin/sh
# Runs from the repo root on Render: `sh backend/start.sh`
cd "$(dirname "$0")"   # -> backend/
PYTHONPATH=../worker:. celery -A celery_app worker --loglevel=info --pool=solo --without-mingle --without-gossip --without-heartbeat &
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
