"""Official Apache-2.0 YOLOX architecture and consistent preprocessing."""
from __future__ import annotations
import os, sys
from pathlib import Path


def add_yolox_source(source=None):
    source=Path(source or os.environ.get("YOLOX_SOURCE_DIR",Path(__file__).parent/"vendor"/"YOLOX")).resolve()
    if not (source/"yolox"/"__init__.py").is_file():
        raise FileNotFoundError("Set YOLOX_SOURCE_DIR to the pinned Megvii YOLOX source directory")
    if str(source) not in sys.path:
        sys.path.insert(0,str(source))
    return source


def build_model(size="s", source=None):
    from common import MODEL_SIZES
    add_yolox_source(source)
    from yolox.exp import Exp
    exp=Exp()
    exp.depth,exp.width=MODEL_SIZES[size]
    exp.num_classes=1
    return exp.get_model()


def load_state(path, device="cpu"):
    import torch
    # Only state dictionaries and primitive metadata are accepted for project checkpoints.
    import numpy as np
    allowed=[np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float32)),type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location=device,weights_only=True)


def warm_start(model,path):
    checkpoint=load_state(path)
    state=checkpoint.get("model",checkpoint)
    target=model.state_dict()
    compatible={k:v for k,v in state.items() if k in target and target[k].shape==v.shape}
    if len(compatible) < len(target)*0.5:
        raise ValueError("The initial checkpoint is incompatible with the selected YOLOX size")
    model.load_state_dict(compatible,strict=False)
    return {"transferred_tensors":len(compatible),"model_tensors":len(target),
            "skipped_tensors":len(target)-len(compatible)}


def preprocessing(image_bgr, size):
    from yolox.data.data_augment import preproc
    return preproc(image_bgr,(size,size))


def postprocessing(outputs, confidence, nms_threshold):
    from yolox.utils import postprocess
    return postprocess(outputs,1,confidence,nms_threshold,class_agnostic=True)
