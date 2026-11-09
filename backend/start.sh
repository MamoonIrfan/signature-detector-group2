  #!/bin/sh
  celery -A worker worker --loglevel=info --pool=solo --without-mingle --without-gossip --without-heartbeat &
  exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}