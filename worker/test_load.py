import time
from tasks import submit_job, get_job_status

N_GOOD, N_BAD = 15, 3
start = time.time()

ids = [submit_job(f"img_{i}.jpg") for i in range(N_GOOD)]
ids += [submit_job(f"bad_{i}.jpg") for i in range(N_BAD)]
print(f"{len(ids)} jobs submit ho gayi ({N_GOOD} theek, {N_BAD} kharab)")

pending = set(ids)
while pending and time.time() - start < 300:
    for j in list(pending):
        if get_job_status(j)["status"] in ("done", "failed"):
            pending.discard(j)
    print(f"{time.strftime('%H:%M:%S')}  baqi jobs: {len(pending)}")
    time.sleep(5)

done = sum(get_job_status(j)["status"] == "done" for j in ids)
failed = sum(get_job_status(j)["status"] == "failed" for j in ids)
print(f"\nResult: done={done}, failed={failed}, total time={time.time() - start:.0f}s")
