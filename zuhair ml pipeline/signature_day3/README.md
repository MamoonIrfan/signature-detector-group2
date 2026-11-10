# Signature detection: Day 3

Revision 1 fixes an omitted extensionless LICENSE in the initial package and
preserves it during smoke handoff repacking. If the initial package failed after
epoch 2 with missing `/content/signature_day3/LICENSE`, paste the contents of
`recover_cpu_smoke_license.py` into the existing Colab session, then rerun only
Section 4. It uses the original saved run ID and `--resume`; it retains completed
epochs and retries export/evaluation/upload. Do not rerun the old source-creation
cell after applying this live patch because its source comparison will detect it.

One non-interactive command downloads HF data, trains YOLOX-S, backs up every
epoch to R2, exports ONNX, verifies PyTorch parity, measures metrics, logs MLflow
artifacts and uploads a candidate. The validation gate controls promotion.
`detect.py` keeps its existing image-bytes contract. No Ultralytics is used.

## 1. Prepare credentials and the old model

Use the newly rotated R2 credentials in Colab Secrets or environment variables:

| Name | Value |
|---|---|
| `R2_ACCOUNT_ID` | Existing Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | Replacement R2 access key |
| `R2_SECRET_ACCESS_KEY` | Replacement R2 secret |
| `R2_BUCKET` | `signature-detection` |
| `HF_TOKEN` | Optional, if required for HF access |

Use Object Read & Write permission scoped to the bucket. Keep keys out of Git,
notebook code and frontend code. The supplied R2 client uses the default
jurisdiction endpoint; jurisdictional buckets require that endpoint to be adapted.

Before FULL training, complete Day 2 Section 13. This baseline manifest must exist:

```
signature-detection-day2/runs/signature_day2_fresh30_b16_20261108/model/manifest.json
```

It must reference final trained ONNX weights, matching metadata and a passing
parity report with SHA-256 records. A missing baseline stops the full run before
training. An early integration export is rejected. CPU smoke needs no baseline.

## 2. Start with the supplied Colab workbook

Extract `Signature_YOLOX_S_Day3_Package.zip` on your computer. Upload
`Signature_YOLOX_S_Day3_Colab.ipynb` through Colab > File > Upload notebook.
The notebook embeds all sources, so you do not need to upload the package again.
Run cells in order. The initial profile is `cpu_smoke`:

| Profile | Epochs | Batch | Data | Can promote? |
|---|---:|---:|---|---|
| `cpu_smoke` | 2 | 2 | 32 train, 16 val, 16 test images | Never |
| `full_gpu` | 30 | 16 | All cleaned examples | Only if gate passes |

Archives may require a full download even in smoke mode; only selected images are
extracted, decoded and trained on. Smoke metrics are not final model metrics.
Weak smoke weights can fail positive-image acceptance. That failure is recorded
and does not turn a smoke-completed run into a deployment-ready model.

## 3. One-command terminal use

Run from the `signature_day3` directory on Linux (Colab, Kaggle with Internet,
another permitted GPU host, or WSL2). The outer Python needs pip. The command
bootstraps an isolated Python 3.10, torch 2.6.0, torchvision 0.21.0 and pinned
requirements including MLflow 2.22.0. It never replaces the notebook's packages.
YOLOX is pinned to `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`.

Full training, after injecting environment secrets:

```bash
python train_day3.py --dataset AimaArshad/signature-detection-yolox --dataset-revision main --epochs 30 --batch-size 16 --device cuda --output outputs/day3_gpu_001
```

CPU integration fallback:

```bash
python train_day3.py --dataset AimaArshad/signature-detection-yolox --epochs 2 --batch-size 2 --device cpu --workers 0 --smoke --output outputs/day3_cpu_001
```

The output basename is also the R2 run ID: choose a NEW name for every fresh run.
`main` resolves to a concrete HF commit; pass an immutable SHA for repeatability.
The default revision is cleaned Day 2 SHA `a55352328a1fbc3442b54db0e377f5add470bcba`.
New runs initialize from official COCO weights. `--initial-checkpoint` optionally
accepts a local checkpoint for a new warm start; this differs from `--resume`.

Print the plan without installing or downloading:

```bash
python train_day3.py --epochs 2 --device cpu --smoke --output outputs/plan_only --dry-run
```

## 4. MLflow and the evaluation gate

No Comet account or hosted MLflow service is needed. Real MLflow tracking and
Model Registry use SQLite metadata with R2-backed artifact URIs. A consistent
SQLite snapshot is uploaded at startup, after every epoch and at completion.
The next writer restores it, preserving registered versions across runtimes.
This is a single-writer course-project setup, not a concurrent MLflow service.

Logged items include effective params, dataset audit, epoch loss/mAP/precision/
recall, final ONNX metrics, the ONNX model, parity report, crops, error examples,
gate report and upload receipt. The original per-epoch full-state R2 checkpoint
backup remains active. Dataset license information comes from its actual HF card.

On the first full run the existing Day 2 model is imported as a registry version,
explicitly tagged as an import. Both baseline and candidate are evaluated on
exactly the SAME current validation examples with the same postprocessing:

```
candidate ONNX validation mAP50 > baseline ONNX validation mAP50 + min_delta
```

`min_delta` defaults to 0.0. Equal/lower scores reject. No historical epoch-7 AP is
hardcoded. Test AP is reported, not compared for promotion. Thresholds stay at
confidence 0.35, NMS 0.45 and AP candidates 0.001. Full training, passing parity
and positive/negative image acceptance are also required.

The full registry is `signature-yolox-s`; improved candidates get a new model
version and `champion` alias. Rejected candidates remain logged/uploaded but do
not replace the champion. Smoke uses `signature-yolox-s-smoke` and never promotes.

Deployment reads this authoritative R2 pointer:

```
signature-detection-day2/registry/signature-yolox-s/champion.json
```

Farhan's backend should download the ONNX and metadata using the pointer's hash
records, then reload its detector worker. This package updates the registry
pointer; it does not automatically reload an already running backend.
If a final alias update is interrupted, the next run reconciles it from the R2
pointer. Inspect the pointer after an interrupted promotion before retrying.

## 5. Inspect the result

Under the selected output directory:

- `reports/day3_status.json`: real execution outcome and acceptance status.
- `reports/gate.json`: baseline/candidate AP and exact decision reasons.
- `reports/metrics_table.csv` and `.png`: measured metrics table.
- `reports/upload_receipt.json`: R2 model records and version if promoted.
- `artifact/farhan_handoff.zip`: matching ONNX, metadata, detector and licenses.
- `acceptance/acceptance_tests.json`: actual signed/unsigned image results.
- `reports/ERRORS.md`, `errors.csv`, optional `error_examples.png`.

| Status | Meaning |
|---|---|
| `promoted` | Improvement and checks passed; registry pointer updated |
| `rejected` | Candidate measured/uploaded; previous champion retained |
| `smoke_completed` | CPU trial finished; production promotion disabled |
| `failed` | Execution or a required check failed; inspect traceback/evidence |

An AP rejection is a valid gate result, not an execution failure. Full-candidate
image acceptance failure returns a failing exit code after retaining evidence.
Smoke acceptance failures remain clearly recorded in its JSON status.

The notebook includes an optional MLflow review UI cell. For Linux terminal use:

```bash
.runtime-py310-cpu/bin/python restore_tracking.py --registry-name signature-yolox-s-smoke --serve
```

Open `http://127.0.0.1:5000` on the same local Linux/WSL host. For full GPU runs use
the CUDA environment and registry name `signature-yolox-s`. Inject the same R2
secrets to load private artifacts. The review copy never writes back to R2.

## 6. Set up GitHub Actions for the remote CPU test

Standard `ubuntu-latest` is a CPU runner, not a free GPU. Included compute minutes
and storage depend on the GitHub account/repository. A permitted GPU runtime can
run the full command; the supplied Actions workflow is the explicit CPU fallback.

Copy sources into your ACTUAL GitHub repository as:

```
signature_day3/train_day3.py
signature_day3/day3_registry.py
signature_day3/requirements_training.txt
signature_day3/...remaining source files and fixtures...
.github/workflows/signature-day3.yml
```

Copy the ZIP's `github_workflow/signature-day3.yml` to the repository-root
`.github/workflows/` directory. Commit sources/workflow to the default branch,
usually `main`. Do not commit `.env`, environments, datasets, downloads or outputs.

GitHub repository > Settings > Secrets and variables > Actions > New repository
secret: add the four R2 names from step 1 and optional `HF_TOKEN`.
Actions > Signature Day 3 CPU trial > Run workflow: keep epochs 2 and pinned SHA.
The job validates inputs, trains on the bounded subset and backs up epochs and
MLflow to R2. Reports/handoff files are also uploaded as Actions artifacts.

## 7. Test Farhan's trigger

With an authenticated GitHub CLI:

```bash
gh workflow run signature-day3.yml --repo OWNER/REPOSITORY --ref main -f epochs=2 -f dataset_revision=a55352328a1fbc3442b54db0e377f5add470bcba
```

For the backend, create a fine-grained GitHub token restricted to that repository
with Actions write permission. Store it only as the backend environment secret
`GH_TRAINING_TOKEN`. GitHub App installation tokens are also suitable.

```bash
python trigger_remote.py --repo OWNER/REPOSITORY --epochs 2
```

Or call from Farhan's authenticated backend event handler:

```python
from trigger_remote import dispatch
result = dispatch('OWNER/REPOSITORY',
    revision='a55352328a1fbc3442b54db0e377f5add470bcba', epochs=2)
```

Replace OWNER/REPOSITORY with the real repository. `trigger_accepted` only means
queued. Wait for the actual job conclusion and inspect its artifacts/status.
Save the run URL, completion screenshot, R2 candidate path, MLflow run ID and
gate/metrics reports as today's evidence. This package supplies the remote
trigger; it does not create Farhan's API endpoint or register an HF webhook.

## 8. Resume after disconnects

Restart the same command with `--resume`, keeping output basename, dataset,
TOTAL target epochs and batch size. Device may change, but full CPU training with
batch 16 can be slow and memory-intensive. CPU smoke is a separate new run.

```bash
python train_day3.py --dataset AimaArshad/signature-detection-yolox --dataset-revision main --epochs 30 --batch-size 16 --device cuda --output outputs/day3_gpu_001 --resume
```

Only one writer may use a registry. An atomic R2 writer lock protects shared
SQLite state and the champion pointer for the run. A hard runtime kill can leave
`registry/<registry-name>/writer_lock.json`. Verify the named old process stopped,
then delete only that stale lock in R2 before resuming. Locks do not auto-expire.
Never remove a live writer's lock. After a failure before epoch 1 there is no full
checkpoint: use a new run ID. Keep Day 2 as baseline; use a NEW ID for Day 3.

## Validation and references

See `DAY3_VALIDATION.md` for actual offline checks and limits. Cloud installs,
live credentials, training/parity/metrics and a real remote trigger still need
execution in your connected runtime. No new model score is claimed here.

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.onnx.html
- https://mlflow.org/docs/latest/ml/model-registry/workflow
- https://mlflow.org/docs/latest/tracking/backend-stores/
- https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event
- https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-running-a-workflow
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job
- https://developers.cloudflare.com/r2/api/s3/api/
