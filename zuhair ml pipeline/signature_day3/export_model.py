"""Official ONNX export, real-image parity, and the small Farhan handoff."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
from common import read_json, write_json, sha256_file, YOLOX_COMMIT

ROOT = Path(__file__).resolve().parent
HANDOFF_SOURCE_FILES = ('detect.py', 'requirements_inference.txt', 'FARHAN_HANDOFF.md', 'LICENSE', 'NOTICE.txt')
HANDOFF_FILES = ('signature_yolox_s.onnx', 'metadata.json', 'onnx_parity.json', 'detect.py',
                 'requirements_inference.txt', 'FARHAN_HANDOFF.md', 'LICENSE', 'NOTICE.txt', 'LICENSE-YOLOX.txt')


def validate_handoff_sources(root=ROOT):
    root = Path(root)
    missing = [name for name in HANDOFF_SOURCE_FILES if not (root/name).is_file()]
    if missing:
        raise FileNotFoundError('Handoff source assets missing: '+', '.join(missing)+
                                '. Restore the complete project package before training/export.')


def package_handoff(artifact):
    artifact = Path(artifact)
    missing = [name for name in HANDOFF_FILES if not (artifact/name).is_file()]
    if missing:
        raise FileNotFoundError('Handoff artifacts missing: '+', '.join(missing))
    bundle = artifact/'farhan_handoff.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in HANDOFF_FILES:
            archive.write(artifact/name, name)
    return bundle


def verify(checkpoint, onnx_path, dataset, output_dir, confidence, nms_iou):
    import cv2
    import onnx
    import onnxruntime as ort
    import torch
    from modeling import build_model, load_state
    from detect import preprocess, postprocess
    from geometry import iou_xyxy
    onnx.checker.check_model(str(onnx_path))
    model = build_model('s').cpu().eval()
    model.load_state_dict(load_state(checkpoint)['model'], strict=True)
    model.head.decode_in_inference = False
    # Exporter replaces nn.SiLU with this algebraically identical module.
    from torch import nn
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module
    model = replace_module(model, nn.SiLU, SiLU)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    val = read_json(Path(dataset)/'annotations'/'instances_val2017.json')
    rows = sorted(val['images'], key=lambda row: str(row['file_name']))[:2]
    samples = []
    for row in rows:
        path = Path(dataset)/'val2017'/row['file_name']
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode parity image: '+str(path))
        samples.append((row['file_name'], image))
    samples.append(('synthetic_white_negative', np.full((800, 600, 3), 255, dtype=np.uint8)))
    report = {'passed': False, 'rtol': 1e-3, 'atol': 1e-3,
              'bbox_tolerance_pixels': 0.5, 'score_tolerance': 1e-3,
              'raw_shape': [1, 8400, 6], 'decode_in_graph': False, 'samples': []}
    destination = Path(output_dir)/'onnx_parity.json'
    try:
        for name, image in samples:
            tensor, ratio = preprocess(image)
            with torch.inference_mode():
                expected = model(torch.from_numpy(tensor)).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            if actual.shape != (1, 8400, 6):
                raise AssertionError('Unexpected ONNX output shape')
            case = {'image': name, 'max_abs_error': float(np.max(np.abs(actual-expected)))}
            report['samples'].append(case)
            np.testing.assert_allclose(actual, expected, rtol=report['rtol'], atol=report['atol'])
            h, w = image.shape[:2]
            left = postprocess(expected, ratio, w, h, confidence, nms_iou)
            right = postprocess(actual, ratio, w, h, confidence, nms_iou)
            if len(left) != len(right):
                raise AssertionError(f'Post-NMS count mismatch on {name}: {len(left)} vs {len(right)}')
            used = set()
            for reference in left:
                options = [(iou_xyxy(reference['bbox'], candidate['bbox']), index)
                           for index, candidate in enumerate(right) if index not in used]
                overlap, index = max(options, default=(0.0, -1))
                if index < 0 or overlap < 0.99:
                    raise AssertionError('Post-NMS detection could not be matched at IoU>=0.99')
                used.add(index)
                np.testing.assert_allclose(reference['bbox'], right[index]['bbox'], rtol=0, atol=0.5)
                if abs(reference['score']-right[index]['score']) > report['score_tolerance']:
                    raise AssertionError('Detection score mismatch')
            case['detections_matched'] = len(left)
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        write_json(destination, report)
        raise
    write_json(destination, report)
    return report


def export_and_verify(checkpoint, dataset, artifact, config, audit, out):
    validate_handoff_sources()
    import onnx
    from modeling import load_state
    artifact = Path(artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    output = artifact/'signature_yolox_s.onnx'
    # A failed new export must not leave a stale model that looks successful.
    temporary = artifact/'signature_yolox_s.pending.onnx'
    temporary.unlink(missing_ok=True)
    command = [sys.executable, str(ROOT/'run_official.py'), 'export_onnx',
               '-f', str(ROOT/'exp_signature_yolox_s.py'), '-c', str(Path(checkpoint).resolve()),
               '--output-name', str(temporary), '--no-onnxsim', '-o', '11']
    # No --decode_in_inference flag: decode belongs to detect.py.
    subprocess.run(command, check=True, cwd=ROOT, env=os.environ.copy())
    if not temporary.is_file():
        raise RuntimeError('Official tools/export_onnx.py did not produce the ONNX file')
    model = onnx.load(str(temporary))
    onnx.helper.set_model_props(model, {'architecture': 'YOLOX-S', 'num_classes': '1',
                            'decode_in_graph': 'false', 'input_range': '0..255',
                            'channel_order': 'BGR', 'padding': 'top_left_114'})
    onnx.save(model, str(temporary))
    onnx.checker.check_model(str(temporary))
    parity = verify(checkpoint, temporary, dataset, artifact, config['conf_threshold'], config['nms_threshold'])
    os.replace(temporary, output)
    checkpoint_state = load_state(checkpoint)
    training_identity = checkpoint_state.get('identity', {})
    is_this_training = (training_identity.get('schema') == 'day2-v1' and
                        training_identity.get('dataset_revision') == audit['revision'])
    metadata = {'schema_version': 1, 'architecture': 'YOLOX-S', 'num_classes': 1,
                'input_size': 640, 'input_shape': [1, 3, 640, 640], 'output_shape': [1, 8400, 6],
                'decode_in_graph': False, 'strides': [8, 16, 32],
                'preprocessing': {'color': 'BGR', 'layout': 'NCHW', 'dtype': 'float32',
                                  'range': [0, 255], 'divide_by_255': False,
                                  'padding_value': 114, 'padding_position': 'right_bottom'},
                'confidence_threshold': config['conf_threshold'], 'nms_threshold': config['nms_threshold'],
                'bbox_format': 'xyxy; original EXIF-oriented image pixels; x2/y2 exclusive',
                'model_sha256': sha256_file(output), 'checkpoint_sha256': sha256_file(checkpoint),
                'epoch': checkpoint_state.get('epoch', checkpoint_state.get('next_epoch')),
                'run_id': config['run_id'], 'target_epochs': config['epochs'],
                'dataset_repo': audit['repo'], 'dataset_revision': audit['revision'],
                'dataset_license': audit['dataset_license'], 'yolox_commit': YOLOX_COMMIT,
                'checkpoint_training_identity': training_identity,
                'trained_on_selected_snapshot_verified': is_this_training,
                'onnx_parity_passed': parity['passed'],
                'validation_metrics_at_checkpoint': checkpoint_state.get('validation'),
                'training_complete': (is_this_training and
                                      read_json(Path(out)/'reports'/'progress.json')['epoch'] >= config['epochs'])
                                     if (Path(out)/'reports'/'progress.json').is_file() else False}
    write_json(artifact/'metadata.json', metadata)
    for name in HANDOFF_SOURCE_FILES:
        shutil.copy2(ROOT/name, artifact/name)
    source_license = Path(config['source_dir'])/'LICENSE'
    shutil.copy2(source_license, artifact/'LICENSE-YOLOX.txt')
    package_handoff(artifact)
    print('ONNX checker and PyTorch/ONNX raw + post-NMS parity passed.', flush=True)
    return artifact
