\# Worker (Redis + Celery job queue)



Job flow: queued -> processing -> done (or failed)



\## Environment variable (required)

REDIS\_URL = rediss://default:<password>@<host>:6379?ssl\_cert\_reqs=CERT\_REQUIRED



Get it from an Upstash Redis database (free plan): Database -> redis-py tab.

Never commit the real value. Set it in .env locally, or in the

"Environment Variables" section of your deployment platform.



\## Run locally

pip install -r requirements.txt

celery -A celery\_app worker --loglevel=info --pool=solo

(on Linux/Docker, drop --pool=solo)



\## Test

python test\_flow.py

