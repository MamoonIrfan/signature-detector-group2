"""Geometry shared by evaluation, image prediction and PDF-page prediction."""
from __future__ import annotations
import math


def map_box(box, ratio, width, height, padding=8):
    if ratio <= 0 or padding < 0 or min(width, height) <= 0:
        raise ValueError("Invalid image scale, dimensions or padding")
    values = [float(v) for v in box]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid predicted box")
    x1, y1, x2, y2 = [v / ratio for v in values]
    x1, y1 = max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1)))
    x2, y2 = max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    box = [x1, y1, x2, y2]
    crop = [max(0, x1-padding), max(0, y1-padding), min(width, x2+padding), min(height, y2+padding)]
    return box, crop


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.0


def operating_metrics(predictions, annotations, threshold=0.25, match_iou=0.5):
    from collections import defaultdict
    ground_truth, proposed = defaultdict(list), defaultdict(list)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        ground_truth[ann["image_id"]].append([x,y,x+w,y+h])
    for pred in predictions:
        if pred["score"] >= threshold:
            proposed[pred["image_id"]].append(pred)
    tp = fp = 0
    for image_id, rows in proposed.items():
        used = set()
        for pred in sorted(rows, key=lambda p:p["score"], reverse=True):
            x,y,w,h = pred["bbox"]
            candidates = [(iou_xyxy([x,y,x+w,y+h], box), j) for j,box in enumerate(ground_truth[image_id]) if j not in used]
            overlap, index = max(candidates, default=(0.0,-1))
            if index >= 0 and overlap >= match_iou:
                used.add(index); tp += 1
            else:
                fp += 1
    fn = len(annotations) - tp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {"precision":precision, "recall":recall, "f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "tp":tp, "fp":fp, "fn":fn, "confidence_threshold":threshold, "matching_iou":match_iou}
