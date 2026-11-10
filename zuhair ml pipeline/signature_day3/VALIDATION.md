# Preparation validation — 08 November 2026

- 34 offline code tests passed on Python 3.12.4, using installed NumPy/Pillow.
- All project Python files compile. Every notebook code cell parses as Python.
- The notebook is nbformat 4.5 with empty outputs and unique cell IDs.
- Embedded source files and the ZIP are checked against the delivered source;
  the unsigned-image fixture is generated without any signature.

Covered: grid/stride decode at all three scales, NMS duplicates, objectness × class
confidence, padding/clipping, invalid/empty input, original-resolution crop pixels,
unique request paths, raw BGR range, dataset negatives, clipping, orphan/degenerate
labels, exact pixel leakage, one-to-one precision/recall, verified Drive backups,
R2 latest-pointer ordering/failures and checkpoint recovery/revision pinning. Additional checks cover required R2 failure handling, unused Run ID protection, and distinguishing missing R2 objects from permission errors.

These are code tests with controlled model output and fake storage where noted.
They do not establish model accuracy or real cloud connectivity.

This preparation environment has no PyTorch, ONNX Runtime or GPU and could not
fetch the live HF snapshot. The notebook therefore performs its pinned dependency
setup, dataset verification, full GPU training, official ONNX export, real-image
PyTorch/ONNX parity, measured evaluation, actual positive/negative image tests
and authenticated R2/HF model upload when run in Colab. Their outputs are pending;
no trained ONNX file, accuracy value or real-model test pass is claimed here.
