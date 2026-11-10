"""Offline code tests. Real model acceptance is run separately by pipeline.py test."""
import io
import json
import contextlib
import shutil
import uuid
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json, read_json, safe_relative, identity, validate_config
from detect import SignatureDetector, decode_output, postprocess, preprocess, nms
from geometry import operating_metrics
from dataset import validate_coco, audit_local_dataset, extract_images
from checkpoints import backup_epoch, recover
from r2_storage import R2Store
from pipeline import assert_fresh_run_available


@contextlib.contextmanager
def scratch():
    # Normal inherited permissions also work in managed Windows workspaces.
    root = Path(__file__).resolve().parents[1]/'test_scratch'
    root.mkdir(exist_ok=True)
    path = root/uuid.uuid4().hex
    path.mkdir()
    try:
        yield str(path)
    finally:
        target = path.resolve()
        if not target.is_relative_to(root.resolve()) or target == root.resolve():
            raise ValueError('Unsafe test cleanup target')
        shutil.rmtree(target)


def raw_box(empty=False):
    raw = np.zeros((1, 8400, 6), dtype=np.float32)
    if not empty:
        # stride=8, grid=(25,28), center=(200,230), size=(200,60)
        index = 28*80+25
        raw[0, index] = [0, .75, np.log(25), np.log(7.5), .9, .9]
        raw[0, index+1] = [-1, .75, np.log(25), np.log(7.5), .8, .9]
    return raw


class FakeSession:
    def __init__(self, raw):
        self.raw = raw
        self.input = None
    def get_inputs(self):
        return [types.SimpleNamespace(shape=[1, 3, 640, 640], type='tensor(float)', name='images')]
    def run(self, outputs, inputs):
        self.input = inputs['images']
        return [self.raw]


def image_bytes():
    buffer = io.BytesIO()
    Image.new('RGB', (640, 640), (31, 62, 93)).save(buffer, format='PNG')
    return buffer.getvalue()


class InferenceTests(unittest.TestCase):
    def test_decode_grid_and_exponential_size(self):
        result = decode_output(raw_box())
        np.testing.assert_allclose(result[28*80+25, :4], [200, 230, 200, 60], atol=1e-4)

    def test_decode_all_three_stride_offsets(self):
        result = decode_output(np.zeros((1, 8400, 6), dtype=np.float32))
        np.testing.assert_array_equal(result[6400, :4], [0, 0, 16, 16])
        np.testing.assert_array_equal(result[8000, :4], [0, 0, 32, 32])

    def test_no_double_decode(self):
        result = decode_output(raw_box())
        np.testing.assert_array_equal(decode_output(result[None], True), result)

    def test_nms_suppresses_duplicate(self):
        result = postprocess(raw_box(), 1, 640, 640)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]['bbox'], [100, 200, 300, 260], atol=1e-4)
        self.assertAlmostEqual(result[0]['score'], .81, places=5)

    def test_threshold_uses_objectness_times_class(self):
        self.assertEqual(postprocess(raw_box(), 1, 640, 640, .85), [])

    def test_empty_predictions(self):
        self.assertEqual(postprocess(raw_box(True), 1, 640, 640), [])

    def test_bad_output_shape_rejected(self):
        with self.assertRaises(ValueError):
            decode_output(np.zeros((1, 8400, 85), dtype=np.float32))

    def test_padded_region_boxes_discarded(self):
        raw = np.zeros((1, 8400, 6), dtype=np.float32)
        raw[0, 0] = [500, 100, 20, 20, .9, .9]
        self.assertEqual(postprocess(raw, 1, 200, 400, decoded=True), [])

    def test_nonfinite_values_do_not_escape(self):
        raw = raw_box()
        raw[0, 0] = [np.nan, 1, 1, 1, .9, .9]
        self.assertEqual(len(postprocess(raw, 1, 640, 640)), 1)

    def test_top_left_letterbox_bgr_and_raw_range(self):
        image = np.full((100, 200, 3), [12, 34, 56], dtype=np.uint8)
        # Constant-color resize is exact under OpenCV linear interpolation.
        def resize(source, size, interpolation):
            self.assertEqual(size, (640, 320))
            return np.full((size[1], size[0], 3), source[0, 0], dtype=np.uint8)
        fake = types.SimpleNamespace(resize=resize, INTER_LINEAR=1)
        with mock.patch.dict(sys.modules, {'cv2': fake}):
            tensor, ratio = preprocess(image)
        self.assertEqual(ratio, 3.2)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [12, 34, 56])
        np.testing.assert_array_equal(tensor[0, :, 400, 0], [114, 114, 114])

    def test_detect_saves_original_resolution_crop(self):
        with scratch() as temp:
            session = FakeSession(raw_box())
            detector = SignatureDetector('unused.onnx', output_dir=temp, session=session)
            result = detector.detect(image_bytes())
            self.assertEqual(len(result), 1)
            x1, y1, x2, y2 = result[0]['bbox']
            with Image.open(result[0]['crop_path']) as crop:
                self.assertEqual(crop.size, (x2-x1, y2-y1))
                self.assertEqual(crop.getpixel((0, 0)), (31, 62, 93))
            np.testing.assert_array_equal(session.input[0, :, 0, 0], [93, 62, 31])

    def test_unsigned_contract_empty_list_and_no_crop_directory(self):
        with scratch() as temp:
            path = Path(temp)/'crops'
            detector = SignatureDetector('unused.onnx', output_dir=path, session=FakeSession(raw_box(True)))
            self.assertEqual(detector.detect(image_bytes()), [])
            self.assertFalse(path.exists())

    def test_request_crops_do_not_overwrite(self):
        with scratch() as temp:
            detector = SignatureDetector('unused.onnx', output_dir=temp, session=FakeSession(raw_box()))
            left, right = detector.detect(image_bytes()), detector.detect(image_bytes())
            self.assertNotEqual(left[0]['crop_path'], right[0]['crop_path'])

    def test_invalid_input_raises(self):
        detector = SignatureDetector('unused.onnx', session=FakeSession(raw_box()))
        with self.assertRaises(ValueError):
            detector.detect(b'')
        with self.assertRaises(Exception):
            detector.detect(b'not an image')


def fixture(name='image.png', negative=False):
    return {'images': [{'id': 1, 'file_name': name, 'width': 100, 'height': 80}],
            'categories': [{'id': 7, 'name': 'signature'}],
            'annotations': [] if negative else [{'id': 1, 'image_id': 1, 'category_id': 7,
                                                 'bbox': [10, 20, 30, 15], 'iscrowd': 0}]}


class DatasetAndMetricsTests(unittest.TestCase):
    def test_negative_image_is_kept(self):
        coco, _ = validate_coco(fixture(negative=True))
        self.assertEqual(len(coco['images']), 1)
        self.assertEqual(coco['annotations'], [])

    def test_category_id_is_normalized(self):
        coco, _ = validate_coco(fixture())
        self.assertEqual(coco['annotations'][0]['category_id'], 1)

    def test_orphan_annotation_rejected(self):
        raw = fixture()
        raw['annotations'][0]['image_id'] = 99
        with self.assertRaises(ValueError):
            validate_coco(raw)

    def test_degenerate_box_rejected(self):
        raw = fixture()
        raw['annotations'][0]['bbox'][2] = -1
        with self.assertRaises(ValueError):
            validate_coco(raw)

    def test_clip_boxes_and_record_count(self):
        raw = fixture()
        raw['annotations'][0]['bbox'] = [-5, 20, 20, 15]
        coco, count = validate_coco(raw)
        self.assertEqual(count, 1)
        self.assertEqual(coco['annotations'][0]['bbox'], [0, 20, 15, 15])

    def test_decoded_pixel_leakage_detected_after_reencoding(self):
        with scratch() as temp:
            root = Path(temp)
            for split, color in [('train', 'red'), ('val', 'green'), ('test', 'blue')]:
                directory = root/(split+'2017')
                directory.mkdir()
                Image.new('RGB', (100, 80), color).save(directory/'image.png')
                write_json(root/'annotations'/f'instances_{split}2017.json', fixture(negative=split == 'val'))
            audit = audit_local_dataset(root)
            self.assertEqual(audit['splits']['val']['negative_images'], 1)
            Image.new('RGB', (100, 80), 'red').save(root/'val2017'/'image.png', compress_level=0)
            with self.assertRaises(ValueError):
                audit_local_dataset(root)

    def test_zip_traversal_rejected(self):
        with scratch() as temp:
            archive = Path(temp)/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape.png', b'bad')
            with self.assertRaises(ValueError):
                extract_images(archive, Path(temp)/'output', 'train2017')

    def test_unsafe_paths_rejected(self):
        for name in ('../x', 'C:/x', '/x', 'a\\x'):
            with self.assertRaises(ValueError):
                safe_relative(name)

    def test_precision_recall_one_to_one_and_negative(self):
        anns = [{'image_id': 1, 'bbox': [10, 10, 20, 20]}, {'image_id': 3, 'bbox': [0, 0, 10, 10]}]
        preds = [{'image_id': 1, 'bbox': [10, 10, 20, 20], 'score': .9},
                 {'image_id': 1, 'bbox': [10, 10, 20, 20], 'score': .8},
                 {'image_id': 2, 'bbox': [0, 0, 10, 10], 'score': .7}]
        metrics = operating_metrics(preds, anns, .35, .5)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 2, 1))
        self.assertAlmostEqual(metrics['precision'], 1/3)
        self.assertEqual(metrics['recall'], .5)


class FakeStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
    def put_file(self, path, key):
        if self.fail:
            raise RuntimeError('simulated failed upload')
        self.calls.append(key)
        return {'key': key, 'sha256': 'fake'}
    def put_json(self, value, key):
        self.calls.append(key)
        return {'key': key}


class BackupTests(unittest.TestCase):
    def setup_files(self, root):
        out = Path(root)/'run'
        (out/'checkpoints').mkdir(parents=True)
        (out/'checkpoints'/'latest.pth').write_bytes(b'latest training state')
        (out/'checkpoints'/'best.pth').write_bytes(b'best weights')
        write_json(out/'config.json', {'run_id': 'demo'})
        write_json(out/'reports'/'history.json', [{'epoch': 1}])
        write_json(out/'reports'/'progress.json', {'best_epoch': 1})
        return out

    def test_drive_snapshot_hash_and_latest_pointer(self):
        with scratch() as temp:
            out = self.setup_files(temp)
            config = {'drive_dir': str(Path(temp)/'drive'), 'run_id': 'demo'}
            backup_epoch(out, config, 1, {'schema': 'day2-v1'})
            target = Path(config['drive_dir'])/'demo'
            self.assertEqual((target/'epoch_0001.pth').read_bytes(), b'latest training state')
            self.assertEqual(read_json(target/'latest.json')['epoch'], 1)
            self.assertEqual(read_json(out/'reports'/'backup_status.json')['saved_to'], ['Drive'])

    def test_r2_pointer_is_last_and_failure_does_not_commit(self):
        with scratch() as temp:
            out = self.setup_files(temp)
            store = FakeStore()
            backup_epoch(out, {'run_id': 'demo', 'require_r2_backup': True}, 1, {'schema': 'day2-v1'}, store)
            self.assertTrue(store.calls[-1].endswith('latest.json'))
            broken = FakeStore(fail=True)
            with self.assertRaises(RuntimeError):
                backup_epoch(out, {'run_id': 'demo'}, 2, {'schema': 'day2-v1'}, broken)
            self.assertEqual(broken.calls, [])

    def test_drive_survives_r2_failure(self):
        with scratch() as temp:
            out = self.setup_files(temp)
            config = {'run_id': 'demo', 'drive_dir': str(Path(temp)/'drive')}
            backup_epoch(out, config, 1, {'schema': 'day2-v1'}, FakeStore(fail=True))
            status = read_json(out/'reports'/'backup_status.json')
            self.assertEqual(status['saved_to'], ['Drive'])
            self.assertEqual(status['failures'][0]['destination'], 'R2')

    def test_required_r2_failure_stops_even_when_drive_succeeds(self):
        with scratch() as temp:
            out = self.setup_files(temp)
            config = {'run_id': 'demo', 'drive_dir': str(Path(temp)/'drive'), 'require_r2_backup': True}
            with self.assertRaisesRegex(RuntimeError, 'Required R2 checkpoint'):
                backup_epoch(out, config, 1, {'schema': 'day2-v1'}, FakeStore(fail=True))
            self.assertTrue((Path(temp)/'drive'/'demo'/'latest.json').is_file())
            self.assertEqual(read_json(out/'reports'/'backup_status.json')['saved_to'], ['Drive'])

    def test_recovery_uses_newest_drive_checkpoint_and_pins_revision(self):
        with scratch() as temp:
            out = Path(temp)/'local'
            (out/'checkpoints').mkdir(parents=True)
            drive = Path(temp)/'drive'/'demo'
            drive.mkdir(parents=True)
            run_identity = {'schema': 'day2-v1', 'dataset_revision': 'f'*40}
            state = {'training_model': {}, 'model': {}, 'optimizer': {}, 'scaler': {},
                     'identity': run_identity, 'next_epoch': 3, 'ema_updates': 20, 'best_epoch': 2}
            (drive/'epoch_0003.pth').write_bytes(b'state')
            (drive/'best_epoch_0002.pth').write_bytes(b'best')
            from common import sha256_file
            write_json(drive/'latest.json', {'epoch': 3,
                       'latest': {'file': 'epoch_0003.pth', 'sha256': sha256_file(drive/'epoch_0003.pth')},
                       'best': {'file': 'best_epoch_0002.pth', 'sha256': sha256_file(drive/'best_epoch_0002.pth')}})
            fake_modeling = types.SimpleNamespace(load_state=lambda path: state if Path(path).name == 'latest.pth'
                                                 else {'identity': run_identity, 'epoch': 2})
            config = {'run_id': 'demo', 'drive_dir': str(Path(temp)/'drive'), 'dataset_revision': 'main'}
            with mock.patch.dict(sys.modules, {'modeling': fake_modeling}):
                restored = recover(out, config)
            self.assertEqual(restored['next_epoch'], 3)
            self.assertEqual(config['dataset_revision'], 'f'*40)


class FreshRunAndR2Tests(unittest.TestCase):
    def test_fresh_run_rejects_local_checkpoint(self):
        with scratch() as temp:
            out = Path(temp)/'run'
            (out/'checkpoints').mkdir(parents=True)
            (out/'checkpoints'/'latest.pth').write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError, 'local RUN_ID'):
                assert_fresh_run_available(out, {'run_id': 'demo'})

    def test_fresh_run_rejects_drive_checkpoint(self):
        with scratch() as temp:
            drive = Path(temp)/'drive'
            write_json(drive/'demo'/'latest.json', {'epoch': 1})
            with self.assertRaisesRegex(ValueError, 'Drive RUN_ID'):
                assert_fresh_run_available(Path(temp)/'local', {'run_id': 'demo', 'drive_dir': str(drive)})

    def test_fresh_run_rejects_r2_checkpoint(self):
        store = types.SimpleNamespace(exists=lambda key: True)
        with scratch() as temp:
            with self.assertRaisesRegex(ValueError, 'R2 RUN_ID'):
                assert_fresh_run_available(Path(temp)/'local', {'run_id': 'demo'}, store)

    def test_unused_r2_run_is_available(self):
        store = types.SimpleNamespace(exists=lambda key: False)
        with scratch() as temp:
            assert_fresh_run_available(Path(temp)/'local', {'run_id': 'demo'}, store)

    def test_r2_missing_key_is_distinct_from_access_failure(self):
        class Missing(Exception):
            response = {'Error': {'Code': 'NoSuchKey'}, 'ResponseMetadata': {'HTTPStatusCode': 404}}
        class Client:
            def head_object(self, **kwargs):
                raise Missing()
        store = R2Store('test-bucket', client=Client())
        self.assertFalse(store.exists('runs/demo/checkpoints/latest.json'))

    def test_r2_permission_error_does_not_look_like_unused_run(self):
        class Denied(Exception):
            response = {'Error': {'Code': 'AccessDenied'}, 'ResponseMetadata': {'HTTPStatusCode': 403}}
        class Client:
            def head_object(self, **kwargs):
                raise Denied()
        store = R2Store('test-bucket', client=Client())
        with self.assertRaisesRegex(RuntimeError, 'AccessDenied'):
            store.exists('runs/demo/checkpoints/latest.json')


if __name__ == '__main__':
    unittest.main()
