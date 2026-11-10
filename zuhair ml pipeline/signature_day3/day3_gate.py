"""Only validation AP@0.5 selects a candidate. No held-out test score is used."""
from __future__ import annotations

import hashlib
import json
import math


def evaluation_protocol(config, audit):
    details = {
        'schema': 'day3-coco-onnx-v1', 'dataset_repo': audit['repo'],
        'dataset_revision': audit['revision'],
        'val_annotation_sha256': audit['splits']['val']['annotation_sha256'],
        'val_images': audit['splits']['val']['images'],
        'ap_candidate_confidence': 0.001, 'nms_iou': config['nms_threshold'],
        'operating_confidence': config['conf_threshold'], 'matching_iou': 0.5,
        'max_detections': 100, 'input_size': 640,
        'preprocessing': 'BGR-float32-0to255-top-left-pad114',
    }
    digest = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'details': details}


def promotion_decision(candidate, baseline, *, smoke, complete, parity, acceptance, min_delta=0.0):
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError('min_delta must be a finite nonnegative number')
    reasons = []
    if smoke:
        reasons.append('smoke_runs_cannot_promote')
    if not complete:
        reasons.append('training_incomplete')
    if not parity:
        reasons.append('onnx_parity_failed')
    if not acceptance:
        reasons.append('positive_or_negative_acceptance_failed')
    candidate_ap = candidate.get('mAP50')
    baseline_ap = baseline.get('mAP50') if baseline else None
    for name, value in (('candidate', candidate_ap), ('baseline', baseline_ap)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            reasons.append(name+'_mAP50_missing_or_invalid')
    if baseline:
        for name in ('protocol_sha256', 'annotation_sha256', 'images'):
            if candidate.get(name) is None or candidate.get(name) != baseline.get(name):
                reasons.append('incomparable_'+name)
    if not reasons and not candidate_ap > baseline_ap + min_delta:
        reasons.append('validation_mAP50_did_not_improve')
    return {'promote': not reasons, 'reasons': reasons, 'metric': 'validation_mAP50',
            'candidate_mAP50': candidate_ap, 'baseline_mAP50': baseline_ap,
            'min_delta': min_delta, 'comparison': 'candidate > baseline + min_delta',
            'held_out_test_score_used_for_comparison': False}
