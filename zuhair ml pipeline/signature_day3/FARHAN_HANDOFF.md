# Farhan: replace the dummy with detect(image_bytes)

The source file `detect.py` is ready for integration before training finishes.
Actual detections require the exported `signature_yolox_s.onnx` and its matching
`metadata.json`. The training pipeline creates `early_handoff/farhan_handoff.zip`
after epoch 1 (optional), then a final `artifact/farhan_handoff.zip` from the
validation-selected best checkpoint after complete training.

## Install and call

Use Python 3.10 or 3.11. Extract the handoff zip, then:

```bash
python -m pip install -r requirements_inference.txt
```

```python
from pathlib import Path
from detect import configure, detect

# Once during each worker process startup, not inside each request:
configure('signature_yolox_s.onnx', metadata_path='metadata.json',
          output_dir='/tmp/signature_crops')

results = detect(Path('document.jpg').read_bytes())
# [{'bbox': [x1, y1, x2, y2], 'score': 0.91,
#   'crop_path': '/tmp/signature_crops/<unique_request>/signature_000.png'}]
# The score shown above illustrates the schema; it is not a measured prediction.
```

You can set `SIGNATURE_ONNX`, `SIGNATURE_METADATA`, and `SIGNATURE_CROP_DIR`
environment variables instead of calling `configure`. There is one lazy ONNX
Runtime CPU session per process. The worker does not need PyTorch, CUDA, the
training dataset or the YOLOX source tree.

## Contract

| Field | Meaning |
|---|---|
| Input | Encoded image bytes: JPEG/PNG or another Pillow-supported raster |
| bbox | Integer `[x1,y1,x2,y2]`, in the original EXIF-oriented image; x2/y2 exclusive |
| score | Objectness multiplied by signature class probability |
| crop_path | Absolute path to a PNG crop on the worker's filesystem |
| No confident boxes | `[]` |
| Invalid image / missing model / inference error | Exception; record job failure/retry |

The crop uses the same bbox with no added padding, at original resolution.
Each request gets a separate output directory. The caller owns file cleanup and
any subsequent R2 upload; a crop_path is not a public URL. For PDFs, render each
page upstream and call detect on each raster; boxes then refer to that raster.

## Preprocessing and postprocessing

1. Apply EXIF orientation, convert RGB to BGR.
2. Keep aspect ratio; scale longest side to 640 and pad right/bottom with 114.
3. HWC to NCHW, float32, values **0–255**. Do not divide by 255 or use ImageNet mean/std.
4. Raw ONNX output `[1,8400,6]`: grid/stride decode for strides 8,16,32.
5. Confidence = objectness × class; threshold from metadata (default 0.35).
6. NMS IoU from metadata (default 0.45), then undo scale, clip, and crop.

The code also supports metadata `decode_in_graph=true` for an official
`--decode_in_inference` export. Do not change this flag without changing the graph.

## Evidence before marking complete

- `onnx_parity.json`: real validation images and a blank input, raw tensor and post-NMS equivalence.
- `reports/metrics_table.csv` and `.png`: measured ONNX mAP@0.5, precision and recall.
- `acceptance/acceptance_tests.json`: positive sample has an overlapping box and crop; unsigned sample returns `[]`.
- `reports/errors.csv`, `ERRORS.md` and optional `error_examples.png`: observed false positives / false negatives.

An empty prediction is an operating-threshold result, not proof the document has
no signature. Detection locates signatures; it does not verify authenticity.
