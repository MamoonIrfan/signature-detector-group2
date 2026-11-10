"""Execute official tools/export_onnx.py with PyTorch 2.6 compatibility.

The source checkout is immutable. YOLOX's old private torch.onnx._export alias
is adapted in this process only; exporter errors retain a nonzero exit status.
"""
import functools
import os
import runpy
import sys
from pathlib import Path
import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ['YOLOX_SOURCE_DIR'])
sys.path[:0] = [str(ROOT), str(SOURCE)]
logger.catch = functools.partial(logger.catch, reraise=True)
if not hasattr(torch.onnx, '_export'):
    def legacy_export(*args, **kwargs):
        return torch.onnx.export(*args, dynamo=False, **kwargs)
    torch.onnx._export = legacy_export
if len(sys.argv) < 2 or sys.argv[1] != 'export_onnx':
    raise SystemExit('Usage: python run_official.py export_onnx [official flags]')
script = SOURCE/'tools'/'export_onnx.py'
sys.argv = [str(script), *sys.argv[2:]]
allow = [np.core.multiarray.scalar, np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]
with torch.serialization.safe_globals(allow):
    runpy.run_path(str(script), run_name='__main__')
