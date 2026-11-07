import time
from tasks import detect_signature, get_job_status

job = detect_signature.delay("test.jpg")
print("1:", get_job_status(job.id))
time.sleep(2)
print("2:", get_job_status(job.id))
time.sleep(6)
print("3:", get_job_status(job.id))