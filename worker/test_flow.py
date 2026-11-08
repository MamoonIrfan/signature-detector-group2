import time
from tasks import submit_job, get_job_status, r


def wait(job_id, timeout=60):
    seen = None
    start = time.time()
    while time.time() - start < timeout:
        s = get_job_status(job_id)
        key = (s["status"], s.get("attempt"))
        if key != seen:
            print("  ", s)
            seen = key
        if s["status"] in ("done", "failed"):
            return
        time.sleep(0.5)
    print("  timeout")


print("1) Normal job:")
j = submit_job("good.jpg")
wait(j)

print("2) Duplicate (wahi job_id dobara, skip hona chahiye):")
submit_job("good.jpg", job_id=j)
print("  ", get_job_status(j)["status"])

print("3) Galat image (3 retries ke baad failed):")
b = submit_job("bad.jpg")
wait(b)

print("4) failed_jobs list:")
for item in r.lrange("failed_jobs", 0, -1):
    print("  ", item)