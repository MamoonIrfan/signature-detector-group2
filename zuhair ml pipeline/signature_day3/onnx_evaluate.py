"""Measure the deployed ONNX graph, not a different training checkpoint."""
from __future__ import annotations
import time
from pathlib import Path
from common import read_json, write_json, sha256_file
from detect import SignatureDetector
from evaluation import coco_metrics
from geometry import operating_metrics


def evaluate_onnx(artifact, dataset, split, output, config):
    import cv2
    from tqdm.auto import tqdm
    artifact, dataset, output = Path(artifact), Path(dataset), Path(output)
    coco = read_json(dataset/'annotations'/f'instances_{split}2017.json')
    if not coco['annotations']:
        raise ValueError('AP/recall require at least one labeled positive in '+split)
    detector = SignatureDetector(artifact/'signature_yolox_s.onnx',
                                 confidence=config['conf_threshold'], nms_iou=config['nms_threshold'])
    predictions = []
    started = time.perf_counter()
    for row in tqdm(coco['images'], desc='ONNX '+split):
        image = cv2.imread(str(dataset/(split+'2017')/row['file_name']))
        if image is None:
            raise ValueError('Cannot decode '+row['file_name'])
        # COCO AP sweeps confidence. It must not use only confidence >= 0.35 boxes.
        for detection in detector.predict_bgr(image, confidence=0.001):
            x1, y1, x2, y2 = detection['bbox']
            predictions.append({'image_id': row['id'], 'category_id': 1,
                                'bbox': [x1, y1, x2-x1, y2-y1], 'score': detection['score']})
    metrics, summary = coco_metrics(coco, predictions)
    metrics.update(operating_metrics(predictions, coco['annotations'], config['conf_threshold'], 0.5))
    metrics.update(split=split, images=len(coco['images']), backend='onnxruntime-cpu',
                   elapsed_seconds=time.perf_counter()-started, model_sha256=sha256_file(artifact/'signature_yolox_s.onnx'),
                   dataset_revision=config['dataset_revision'],
                   annotation_sha256=sha256_file(dataset/'annotations'/f'instances_{split}2017.json'),
                   ap_candidate_confidence=0.001, nms_threshold=config['nms_threshold'])
    write_json(output/(split+'_metrics.json'), metrics)
    write_json(output/(split+'_predictions.json'), predictions)
    (output/(split+'_coco_summary.txt')).write_text(summary, encoding='utf-8')
    return metrics
