# Day 2 — complete YOLOX-S signature training and ONNX handoff

Open `Signature_YOLOX_S_Day2_Colab.ipynb` in Google Colab and select a T4 GPU.
The notebook embeds every project source file and the unsigned-image fixture;
you do not need to upload this source folder separately.

This workbook is prepared code, **not a record of a completed training run**.
No measured accuracy, trained ONNX artifact or real-model test pass is included
until its GPU cells run. It uses only the cleaned dataset
<https://huggingface.co/datasets/AimaArshad/signature-detection-yolox>.
Live repository access was unavailable in the preparation environment, so its
current commit, license, file layout and counts are resolved and checked in Colab.
An older dataset is never substituted.

## Workflow

| Stage | Output |
|---|---|
| Prepare | Immutable HF dataset SHA, all train/val/test images, COCO audit, box previews |
| Train | 30 epochs by default, batch size 16, all training images; val mAP@0.5 / precision / recall every epoch |
| Backup | Full state after every completed epoch to R2 and Drive by default; verified hashes |
| Early handoff | Epoch 1 best → official export, parity, minimal Farhan zip; training continues |
| Final export | Validation-selected best → official `tools/export_onnx.py`, fixed 640 ONNX |
| Validate | CPU ONNX evaluation on val and held-out test; CSV/PNG metrics, errors and examples |
| Test | Deterministic held-out positive bbox+crop, unsigned input must return `[]` |
| Upload | R2 and/or configured HF model repo; upload receipt after successful checks |

An isolated Python 3.10 environment uses torch 2.6.0 / torchvision 0.21.0.
YOLOX source is pinned to official commit
`419778480ab6ec0590e5d3831b3afb3b46ab2aa3` (tag 0.3.0).
No Ultralytics training package is used. `bootstrap.py` tests forward/backward
and NMS before training. Its small random-input check is a setup test; full
training always uses the complete dataset splits.

## Dataset format checked at runtime

Expected cleaned COCO layout (matching this project's existing dataset loader):

```text
annotations/instances_train2017.json
annotations/instances_val2017.json
annotations/instances_test2017.json
train2017.zip  (contains train2017/<COCO file_name>)
val2017.zip
test2017.zip
```

Loose `train2017/`, `val2017/`, `test2017/` directories are supported too. If the
new snapshot uses a different layout, preparation stops with the missing path.
It does not guess a schema, relabel missing files as negatives, mix revisions,
or fall back to tech4humans data.

The audit checks image decoding and dimensions, one signature category, unique
IDs, orphan labels, finite positive-area boxes, EXIF orientation and exact
decoded-pixel overlap across splits. Slightly out-of-bounds boxes are clipped
and counted. Negative images have an image entry with no annotations and are
preserved. Near-duplicate and template overlap need separate review.

## Starting training and meeting the handoff window

Keep `EPOCHS=30` and `BATCH_SIZE=16` for a full run; there is no early stopping. More epochs do not
guarantee better accuracy. Validation selects the best checkpoint by mAP@0.5.
GPU speed and Colab availability determine duration; a 12–2 PM completion is
not guaranteed. The first epoch's export lets Farhan begin integration while
the remaining epochs run. Its metadata marks training as incomplete.

Fresh training is configured with `RESUME=False`, a new Run ID, and empty
`INITIAL_CHECKPOINT` / `EXPORT_CHECKPOINT` settings. Training begins at epoch 1
using official COCO-pretrained YOLOX-S weights; no earlier signature checkpoint
or optimizer state is loaded. This is a fresh fine-tuning run, rather than random
weight initialization. The batch size is 16 actual images per optimizer step,
with no gradient accumulation; the last batch can be smaller.

For a quick handoff before training, set `EXPORT_CHECKPOINT` to a prior
single-class checkpoint, run Prepare, then Export + parity. The exported file
reflects that checkpoint's training; it is not evidence of cleaned-data training.

## Resume after Colab disconnect

1. Reopen the same notebook, mount the same Drive and recreate the environment.
2. Use the same `RUN_ID`, `EPOCHS`, batch size, image size, seed and thresholds.
3. Set `RESUME=True`. Keep the same Drive path/R2 bucket and prefix.
4. Run Prepare and Train. Recovery pins the original dataset SHA, verifies
   hashes, chooses the newest surviving full checkpoint, then continues at
   the next epoch. The latest backup is retried before another epoch runs.

Full checkpoints include training weights, EMA inference weights, optimizer,
AMP scaler, EMA update count, Python/NumPy/Torch/CUDA RNG, history and identity.
Both Drive and R2 write their latest pointer last. R2 backup is required by default. If an R2 upload fails, training stops even
when the Drive backup succeeds; local/Drive files are retained. Fix the connection
and resume before another epoch. Setting require_r2_backup=false explicitly
allows continuing when at least one enabled destination succeeds.
Work within an unfinished epoch cannot be recovered; epoch-level snapshots
are the recovery boundary. Storage is roughly one full training checkpoint
per epoch plus best weights; allow several GB and check your quota.

`best.pth` is weights for inference/fine-tuning. Resume requires the matching
full Day 2 latest/epoch checkpoint. Changing datasets or training schedules
requires a new run. CPU recovery is supported with `DEVICE='cpu'` but is slow;
GPU-only RNG/scaler state is not applied on CPU.

## Metrics and tests

AP uses standard pycocotools with candidate confidence 0.001, COCO maxDets=100
and configured NMS. `mAP50` is AP@IoU 0.5 for the one signature class.
Precision/recall/F1 use confidence 0.35 by default and one-to-one IoU>=0.5
matching. All scores in CSV/JSON are fractions 0–1; multiply by 100 for percent.
Validation is used for checkpoint selection. The test split is evaluated only
after the model/threshold are fixed. Do not tune confidence on test results.

Parity uses two deterministic validation images and a white input. It compares
raw `[1,8400,6]` tensors with rtol=1e-3, atol=1e-3; then matches post-NMS boxes
at IoU>=0.99, <=0.5 pixel coordinate difference and <=1e-3 score difference.
Any failure prevents a successful export handoff.

Acceptance uses the first filename-sorted held-out positive and requires at
least one prediction with IoU>=0.5 plus a valid original-resolution crop.
The negative is a dataset test image with zero annotations, or your configured
`NEGATIVE_IMAGE` (a visually reviewed unsigned document). If neither is
available, the included synthetic printed unsigned document is used and
clearly labeled synthetic. It must also return `[]`; the code does not force
that outcome. One synthetic test is not a broad false-positive evaluation.
Failing tests save evidence and return a failure, rather than claiming Done.
Tune only on validation, add reviewed negative training examples, and rerun a
new evaluation when necessary.

## Secrets and uploads

Colab Secrets/environment variables only:

| Secret | Use |
|---|---|
| HF_TOKEN | Read private/gated dataset; write model repo for HF upload |
| R2_ACCOUNT_ID | Cloudflare account identifier |
| R2_ACCESS_KEY_ID | R2 S3 API credential |
| R2_SECRET_ACCESS_KEY | R2 S3 API credential |
| R2_BUCKET | Existing bucket name |

`ENABLE_R2=True` and `require_r2_backup=true` are defaults. Add the four R2
secrets and allow notebook access. A read/write marker verifies R2 permissions
before dataset preparation/training. A fresh Run ID is rejected if checkpoints
already exist locally, in Drive, or in R2, to prevent overwriting another run.
The final ONNX model, handoff and reports upload to R2 in the upload stage.
You may additionally set `HF_MODEL_REPO` to a repo you can
write, for example `your-username/signature-yolox-s`. New HF model repos are
created private; existing repo visibility is preserved. No upload can happen
without your credentials. Leave checkpoint paths empty for the configured fresh run.
The final status only says Done after training, parity, image tests and a
recorded R2/HF model upload all succeed.

## Headless commands

```bash
python bootstrap.py --device cuda
PY=/path/to/project/.runtime-py310-cuda/bin/python
$PY pipeline.py prepare --config config.json
$PY pipeline.py train --config config.json
$PY pipeline.py export --config config.json
$PY pipeline.py evaluate --config config.json
$PY pipeline.py test --config config.json
$PY pipeline.py publish --config config.json
# Or all stages, with publish_model=true and a configured upload destination:
$PY pipeline.py all --config config.json
```

Run code tests with `$PY -m unittest discover -s tests -v`. These geometry,
contract, dataset and backup tests are distinct from real-model acceptance.

## References

- Cleaned dataset: <https://huggingface.co/datasets/AimaArshad/signature-detection-yolox>
- Official YOLOX source: <https://github.com/Megvii-BaseDetection/YOLOX/tree/419778480ab6ec0590e5d3831b3afb3b46ab2aa3>
- Official ONNX exporter: <https://github.com/Megvii-BaseDetection/YOLOX/blob/419778480ab6ec0590e5d3831b3afb3b46ab2aa3/tools/export_onnx.py>
- Official preprocessing: <https://github.com/Megvii-BaseDetection/YOLOX/blob/419778480ab6ec0590e5d3831b3afb3b46ab2aa3/yolox/data/data_augment.py>
- ONNX Runtime Python API: <https://onnxruntime.ai/docs/api/python/api_summary.html>
- Hugging Face upload API: <https://huggingface.co/docs/huggingface_hub/guides/upload>
- Cloudflare R2 S3 API: <https://developers.cloudflare.com/r2/api/s3/api/>

The detector source retains its Apache-2.0 license. The dataset's declared
license and card are captured from the pinned snapshot at runtime.
