import time
from tasks import submit_job, get_job_status

j = submit_job("slow.jpg")
print("job:", j)
last = None
while True:
    s = get_job_status(j)
    if s["status"] != last:
        print(time.strftime("%H:%M:%S"), s["status"], s.get("attempt", ""))
        if s["status"] == "processing":
            print(">>> ABHI TERMINAL 1 (celery wala) KILL KARO! 25 second hain <<<")
        last = s["status"]
    if s["status"] in ("done", "failed"):
        break
    time.sleep(1)