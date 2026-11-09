"""Farhan's image-bytes contract. No PyTorch or YOLOX installation is needed.

Configure once with configure('signature_yolox_s.onnx', metadata_path='metadata.json'),
then call detect(image_bytes). Bboxes are integer xyxy in the EXIF-oriented image;
x2/y2 are exclusive. Crops use these same coordinates at original resolution.
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIZE = 640
STRIDES = (8, 16, 32)


def preprocess(image_bgr):
    """Official non-legacy YOLOX: top-left letterbox, BGR, CHW float32, 0..255.

    Normalization is the float32 cast. Do NOT divide by 255 or subtract mean/std.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('Expected an HWC three-channel BGR image')
    height, width = image_bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError('Empty image')
    ratio = min(SIZE / height, SIZE / width)
    target = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    if target == (width, height):
        resized = image_bgr
    else:
        import cv2
        resized = cv2.resize(image_bgr, target, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[:target[1], :target[0]] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, ratio


def decode_output(raw, decoded=False):
    """Raw YOLOX head -> center-x, center-y, width, height, obj, class.

    The default export leaves decode outside ONNX. A metadata flag also supports
    official --decode_in_inference exports without decoding coordinates twice.
    """
    expected = sum((SIZE // stride) ** 2 for stride in STRIDES)
    raw = np.asarray(raw)
    if raw.shape != (1, expected, 6):
        raise ValueError(f'Expected one-class output (1,{expected},6); got {raw.shape}')
    output = raw[0].astype(np.float32, copy=True)
    if decoded:
        return output
    grids, scales = [], []
    for stride in STRIDES:
        side = SIZE // stride
        yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
        grids.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
        scales.append(np.full((side * side, 1), stride, dtype=np.float32))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(scales)
    output[:, :2] = (output[:, :2] + grid) * stride
    with np.errstate(over='ignore', invalid='ignore'):
        output[:, 2:4] = np.exp(output[:, 2:4]) * stride
    return output


def nms(boxes, scores, iou_threshold=0.45, max_detections=100):
    order = np.argsort(-scores, kind='stable')
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        wh = np.maximum(0, np.minimum(boxes[current, 2:], boxes[remaining, 2:]) -
                        np.maximum(boxes[current, :2], boxes[remaining, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        iou = intersection / np.maximum(areas[current] + areas[remaining] - intersection, 1e-9)
        order = remaining[iou <= iou_threshold]
    return keep


def postprocess(raw, ratio, width, height, confidence=0.35, nms_iou=0.45, decoded=False):
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1 or ratio <= 0:
        raise ValueError('Invalid threshold or image scale')
    pred = decode_output(raw, decoded)
    scores = pred[:, 4] * pred[:, 5]
    mask = np.isfinite(pred).all(axis=1) & np.isfinite(scores) & (scores >= confidence)
    mask &= (pred[:, 2] > 0) & (pred[:, 3] > 0)
    pred, scores = pred[mask], scores[mask]
    boxes = np.concatenate((pred[:, :2] - pred[:, 2:4] / 2,
                            pred[:, :2] + pred[:, 2:4] / 2), axis=1)
    results = []
    # NMS runs before clipping, as in official YOLOX.
    for index in nms(boxes, scores, nms_iou):
        box = boxes[index] / ratio
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        results.append({'bbox': [float(x) for x in box], 'score': float(scores[index])})
    return results


class SignatureDetector:
    def __init__(self, model_path, metadata_path=None, output_dir='signature_crops',
                 confidence=None, nms_iou=None, session=None):
        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_name('metadata.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
        if metadata.get('input_size', SIZE) != SIZE or metadata.get('num_classes', 1) != 1:
            raise ValueError('This handoff supports only one-class YOLOX-S at 640')
        self.confidence = float(confidence if confidence is not None else metadata.get('confidence_threshold', 0.35))
        self.nms_iou = float(nms_iou if nms_iou is not None else metadata.get('nms_threshold', 0.45))
        if not 0 <= self.confidence <= 1 or not 0 <= self.nms_iou <= 1:
            raise ValueError('Thresholds must be in [0,1]')
        self.decoded = bool(metadata.get('decode_in_graph', False))
        self.output_dir = Path(output_dir)
        if session is None:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.session = session
        if hasattr(session, 'get_modelmeta'):
            graph = session.get_modelmeta().custom_metadata_map
            flag = graph.get('decode_in_graph')
            if flag is not None and (flag.lower() == 'true') != self.decoded:
                raise ValueError('metadata.json decode flag does not match this ONNX graph')
        spec = session.get_inputs()[0]
        if spec.shape != [1, 3, SIZE, SIZE] or spec.type != 'tensor(float)':
            raise ValueError(f'Expected fixed float32 [1,3,640,640], got {spec.shape} {spec.type}')
        self.input_name = spec.name

    def predict_bgr(self, image_bgr, confidence=None):
        tensor, ratio = preprocess(image_bgr)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        height, width = image_bgr.shape[:2]
        return postprocess(raw, ratio, width, height,
                           self.confidence if confidence is None else confidence,
                           self.nms_iou, self.decoded)

    def detect(self, image_bytes):
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
            raise ValueError('image_bytes must contain an encoded image (PNG/JPEG etc.)')
        with Image.open(io.BytesIO(bytes(image_bytes))) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        predictions = self.predict_bgr(bgr)
        if not predictions:
            return []
        request_dir = self.output_dir / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=False)
        results = []
        for index, prediction in enumerate(predictions):
            x1, y1, x2, y2 = prediction['bbox']
            box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
            crop_path = request_dir / f'signature_{index:03d}.png'
            image.crop(tuple(box)).save(crop_path, format='PNG')
            results.append({'bbox': box, 'score': prediction['score'],
                            'crop_path': str(crop_path.resolve())})
        return results


_default = None
_lock = threading.Lock()


def configure(model_path, metadata_path=None, output_dir='signature_crops', **kwargs):
    """Call once during worker startup; each process keeps one ONNX session."""
    global _default
    detector = SignatureDetector(model_path, metadata_path, output_dir, **kwargs)
    with _lock:
        _default = detector
    return detector


def detect(image_bytes):
    """Return [{bbox: [x1,y1,x2,y2], score: float, crop_path: str}], or [].

    Invalid images/model errors raise; they are never disguised as no detections.
    Crop files are local to this worker. The caller owns retention or R2 upload.
    """
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = SignatureDetector(
                    os.environ.get('SIGNATURE_ONNX', 'signature_yolox_s.onnx'),
                    os.environ.get('SIGNATURE_METADATA') or None,
                    os.environ.get('SIGNATURE_CROP_DIR', 'signature_crops'))
    return _default.detect(image_bytes)
