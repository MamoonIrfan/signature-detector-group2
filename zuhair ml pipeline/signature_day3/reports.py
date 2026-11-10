"""Measured metric tables, observed errors, and image/crop acceptance evidence."""
from __future__ import annotations
import collections
import csv
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from common import read_json, write_json
from geometry import operating_metrics, iou_xyxy
from detect import SignatureDetector

ROOT = Path(__file__).resolve().parent


def annotated_image(source, predictions, annotations=()):
    with Image.open(source) as raw:
        image = raw.convert('RGB')
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size)/300))
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        draw.rectangle((x, y, x+w, y+h), outline='#14803c', width=width)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction['bbox']
        draw.rectangle((x1, y1, x2, y2), outline='#e23c3c', width=width)
        draw.text((x1, max(0, y1-14)), f"signature {prediction['score']:.3f}", fill='#d00000')
    return image


def acceptance_tests(artifact, dataset, out, config):
    artifact, dataset, out = Path(artifact), Path(dataset), Path(out)
    evidence = out/'acceptance'
    evidence.mkdir(parents=True, exist_ok=True)
    coco = read_json(dataset/'annotations'/'instances_test2017.json')
    groups = collections.defaultdict(list)
    for annotation in coco['annotations']:
        groups[annotation['image_id']].append(annotation)
    ordered = sorted(coco['images'], key=lambda row: str(row['file_name']))
    positive = next((row for row in ordered if groups[row['id']]), None)
    if positive is None:
        raise ValueError('A held-out test positive is required')
    positive_path = dataset/'test2017'/positive['file_name']
    detector = SignatureDetector(artifact/'signature_yolox_s.onnx', output_dir=evidence/'crops')
    positive_results = detector.detect(positive_path.read_bytes())
    overlaps = []
    for detection in positive_results:
        for ann in groups[positive['id']]:
            x, y, w, h = ann['bbox']
            overlaps.append(iou_xyxy(detection['bbox'], [x, y, x+w, y+h]))
        with Image.open(detection['crop_path']) as crop:
            x1, y1, x2, y2 = detection['bbox']
            if crop.size != (x2-x1, y2-y1) or min(crop.size) <= 0:
                raise AssertionError('Crop dimensions do not match bbox')
    positive_passed = bool(positive_results and max(overlaps, default=0) >= 0.5)
    annotated_image(positive_path, positive_results, groups[positive['id']]).save(evidence/'positive_bbox.png')
    shutil.copy2(positive_path, evidence/('positive_input'+positive_path.suffix))
    if config.get('negative_image'):
        negative_path = Path(config['negative_image'])
        negative_kind = 'user_reviewed_unsigned_document'
    else:
        negative = next((row for row in ordered if not groups[row['id']]), None)
        if negative is not None:
            negative_path = dataset/'test2017'/negative['file_name']
            negative_kind = 'held_out_dataset_negative'
        else:
            negative_path = ROOT/'fixtures'/'unsigned_document.png'
            negative_kind = 'synthetic_printed_unsigned_document'
    negative_results = detector.detect(negative_path.read_bytes())
    negative_passed = negative_results == []
    shutil.copy2(negative_path, evidence/('negative_input'+negative_path.suffix))
    annotated_image(negative_path, negative_results).save(evidence/'negative_bbox.png')
    report = {'passed': positive_passed and negative_passed,
              'confidence_threshold': detector.confidence,
              'positive': {'passed': positive_passed, 'source': str(positive_path),
                           'detections': positive_results, 'best_gt_iou': max(overlaps, default=0)},
              'negative': {'passed': negative_passed, 'source': str(negative_path), 'source_type': negative_kind,
                           'expected': [], 'actual': negative_results}}
    write_json(evidence/'acceptance_tests.json', report)
    print('Positive bbox + crop:', 'PASS' if positive_passed else 'FAIL', flush=True)
    print('Unsigned image returns []:', 'PASS' if negative_passed else 'FAIL', '|', negative_kind, flush=True)
    if not report['passed']:
        print('Handoff acceptance failed. Inspect acceptance/*.png and errors before marking Done.', flush=True)
    return report


def observed_errors(dataset, out, config, split='test'):
    dataset, out = Path(dataset), Path(out)
    coco = read_json(dataset/'annotations'/f'instances_{split}2017.json')
    predictions = read_json(out/'evaluation'/'onnx'/(split+'_predictions.json'))
    gt, proposed = collections.defaultdict(list), collections.defaultdict(list)
    for ann in coco['annotations']:
        gt[ann['image_id']].append(ann)
    for pred in predictions:
        if pred['score'] >= config['conf_threshold']:
            proposed[pred['image_id']].append(pred)
    rows, visual = [], []
    for image in coco['images']:
        image_id = image['id']
        values = operating_metrics(proposed[image_id], gt[image_id], config['conf_threshold'], 0.5)
        labels = []
        if values['fn']:
            labels.append('missed_signature')
        if values['fp']:
            labels.append('false_positive_on_unsigned_image' if not gt[image_id]
                          else 'false_positive_or_duplicate_or_poor_localization')
        if labels:
            rows.append({'image_id': image_id, 'file_name': image['file_name'], 'errors': '; '.join(labels),
                         **{k: values[k] for k in ('tp', 'fp', 'fn')},
                         'gt_count': len(gt[image_id]), 'prediction_count': len(proposed[image_id])})
            visual.append((values['fp']+values['fn'], image, proposed[image_id], gt[image_id]))
    report_dir = out/'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir/'errors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id', 'file_name', 'errors', 'tp', 'fp', 'fn',
                                                   'gt_count', 'prediction_count'])
        writer.writeheader()
        writer.writerows(rows)
    total = collections.Counter()
    for row in rows:
        for label in row['errors'].split('; '):
            total[label] += 1
    write_json(report_dir/'error_summary.json', {'split': split, 'images_evaluated': len(coco['images']),
               'images_with_errors': len(rows), 'observed_image_counts': dict(total)})
    lines = ['# Observed ONNX errors', '', f'Split: {split}; confidence={config["conf_threshold"]}; matching IoU=0.5.',
             f'Images with measured FP/FN: {len(rows)} / {len(coco["images"])}.', '',
             'See errors.csv for every affected image. Red boxes = predictions; green = ground truth.', '']
    lines += [f'- {key}: {value} images' for key, value in total.items()]
    lines += ['', '## Visual review prompts (hypotheses, not automatically proven causes)', '',
              '- Small signatures reduced to very few pixels at 640.',
              '- Faint ink, low contrast, blur or compression.',
              '- Stamps, printed names, handwriting or form lines mistaken for signatures.',
              '- Rotated pages, dense forms, overlapping signatures or unfamiliar templates.',
              '', 'Confirm these causes by inspecting examples; do not label them measured failure modes without evidence.']
    (report_dir/'ERRORS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    panels = []
    for index, (_, image, preds, anns) in enumerate(sorted(visual, key=lambda item: -item[0])[:6]):
        display = [{'bbox': [p['bbox'][0], p['bbox'][1], p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]],
                    'score': p['score']} for p in preds]
        panel = annotated_image(dataset/(split+'2017')/image['file_name'], display, anns)
        panel.thumbnail((480, 520))
        canvas = Image.new('RGB', (500, 560), 'white')
        canvas.paste(panel, ((500-panel.width)//2, 28))
        ImageDraw.Draw(canvas).text((10, 8), f"{index+1}: {str(image['file_name'])[:52]}", fill='black')
        panels.append(canvas)
    if panels:
        montage = Image.new('RGB', (1000, 560*math_ceil_half(len(panels))), '#e8edf3')
        for index, panel in enumerate(panels):
            montage.paste(panel, ((index % 2)*500, (index//2)*560))
        montage.save(report_dir/'error_examples.png')
    return rows


def math_ceil_half(value):
    return (value+1)//2


def metric_tables(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    directory = out/'reports'
    rows = []
    for split in ('val', 'test'):
        metrics = read_json(out/'evaluation'/'onnx'/(split+'_metrics.json'))
        rows.append({'split': split, **{key: metrics[key] for key in
                    ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'images', 'confidence_threshold')}})
    with (directory/'metrics_table.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = ['Split', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']
    cells = [[row['split']]+[f"{row[k]:.4f}" for k in ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1')]
             +[str(row[k]) for k in ('tp', 'fp', 'fn')] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    fig.patch.set_facecolor('#f5f7fb')
    ax.axis('off')
    ax.set_title('YOLOX-S | measured ONNX Runtime metrics', loc='left', fontsize=16, fontweight='bold', pad=14)
    table = ax.table(cellText=cells, colLabels=labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#162b4d' if r == 0 else ('#e6edf8' if r % 2 else '#ffffff'))
        if r == 0:
            cell.get_text().set_color('white')
    confidence = rows[0]['confidence_threshold']
    fig.text(0.02, 0.05, f'Precision/recall: confidence >= {confidence}, IoU >= 0.5. AP candidates >= 0.001. Scores are fractions 0–1.', fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(directory/'metrics_table.png', dpi=180)
    plt.close(fig)
    history_path = directory/'history.json'
    if history_path.is_file():
        history = read_json(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].plot([r['epoch'] for r in history], [r['loss'] for r in history], color='#2563eb')
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training loss')
        for key, label in (('mAP50', 'mAP@0.5'), ('precision', 'Precision'), ('recall', 'Recall')):
            axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
        axes[1].set(xlabel='Epoch', ylabel='Score', title='Validation each epoch', ylim=(0, 1))
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(directory/'learning_curves.png', dpi=160)
        plt.close(fig)
    return rows
