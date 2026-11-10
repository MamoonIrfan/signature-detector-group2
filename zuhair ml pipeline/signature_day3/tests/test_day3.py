"""Meaningful offline checks; no assertion that a real trained model passed."""
import copy
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day3_data import select_subset
from day3_gate import promotion_decision, evaluation_protocol
from day3_registry import RegistryLock
from train_day3 import parser, make_config, download_baseline
from trigger_remote import dispatch


def metrics(ap):
    return {'mAP50': ap, 'protocol_sha256': 'same-protocol',
            'annotation_sha256': 'same-labels', 'images': 16}


def gate(candidate=None, baseline=None, **overrides):
    kwargs = dict(smoke=False, complete=True, parity=True, acceptance=True)
    kwargs.update(overrides)
    return promotion_decision(candidate or metrics(.91), baseline or metrics(.90), **kwargs)


class GateTests(unittest.TestCase):
    def test_strictly_better_validation_promotes(self):
        self.assertTrue(gate()['promote'])

    def test_equal_score_does_not_promote(self):
        self.assertFalse(gate(metrics(.9), metrics(.9))['promote'])

    def test_regression_does_not_promote(self):
        self.assertFalse(gate(metrics(.89))['promote'])

    def test_margin_requires_strictly_more_than_delta(self):
        self.assertFalse(gate(metrics(.75), metrics(.5), min_delta=.25)['promote'])

    def test_smoke_run_cannot_replace_good_model_even_at_perfect_ap(self):
        self.assertFalse(gate(metrics(1), smoke=True)['promote'])

    def test_mismatched_labels_protocol_or_split_size_fail_closed(self):
        for key, value in (('annotation_sha256','different'),('protocol_sha256','different'),('images',15)):
            with self.subTest(key=key):
                changed = metrics(.99)
                changed[key] = value
                self.assertFalse(gate(changed)['promote'])

    def test_missing_protocol_does_not_count_as_matching(self):
        a, b = metrics(.99), metrics(.9)
        a.pop('protocol_sha256'); b.pop('protocol_sha256')
        self.assertFalse(gate(a,b)['promote'])

    def test_invalid_scores_fail_closed(self):
        for value in (None,float('nan'),float('inf'),-.01,1.01,True,'0.99'):
            with self.subTest(value=value):
                self.assertFalse(gate(metrics(value))['promote'])

    def test_no_baseline_is_not_automatic_promotion(self):
        self.assertFalse(promotion_decision(metrics(.99), None, smoke=False,
            complete=True, parity=True, acceptance=True)['promote'])

    def test_each_required_completion_check_blocks_promotion(self):
        for key in ('complete','parity','acceptance'):
            with self.subTest(key=key):
                self.assertFalse(gate(**{key:False})['promote'])

    def test_test_ap_cannot_affect_selection(self):
        candidate = metrics(.89)
        candidate['test_mAP50'] = 1.0
        self.assertFalse(gate(candidate)['promote'])

    def test_invalid_delta_rejected(self):
        for value in (-.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                gate(min_delta=value)

    def test_protocol_changes_when_postprocessing_or_data_changes(self):
        config = {'nms_threshold': .45,'conf_threshold':.35}
        audit = {'repo':'a/b','revision':'sha','splits':{'val':{'annotation_sha256':'labels','images':16}}}
        first = evaluation_protocol(config,audit)['sha256']
        for conf, report in (({**config,'nms_threshold':.5},audit),
                             (config,{**audit,'revision':'newsha'})):
            self.assertNotEqual(first,evaluation_protocol(conf,report)['sha256'])


class DataAndCliTests(unittest.TestCase):
    def coco(self):
        return {'images':[{'id':i,'file_name':f'{i:03}.png'} for i in range(20)],
                'annotations':[{'id':1,'image_id':19,'bbox':[1,2,3,4]}],
                'categories':[{'id':1,'name':'signature'}]}

    def test_small_subset_keeps_a_positive_and_unsigned_sample(self):
        result = select_subset(self.coco(), 2)
        self.assertEqual(len(result['images']),2)
        self.assertIn(19,[row['id'] for row in result['images']])
        self.assertEqual(len(result['annotations']),1)
        self.assertEqual(result,select_subset(self.coco(),2))

    def test_subset_excludes_orphan_annotations(self):
        data = self.coco()
        data['annotations'] += [{'id':2,'image_id':18,'bbox':[1,2,3,4]}]
        result = select_subset(data, 2)
        ids = {row['id'] for row in result['images']}
        self.assertTrue(all(row['image_id'] in ids for row in result['annotations']))

    def test_no_labeled_positive_is_rejected(self):
        data=self.coco();data['annotations']=[]
        with self.assertRaises(ValueError):
            select_subset(data,2)

    def test_full_default_batch_is_16_for_30_epochs(self):
        config,_ = make_config(parser().parse_args(['--output','outputs/check-full']))
        self.assertEqual((config['batch_size'],config['epochs']),(16,30))
        self.assertFalse(config['smoke'])

    def test_cpu_trial_has_separate_registry_and_small_batch(self):
        config,_ = make_config(parser().parse_args(['--output','outputs/check-smoke','--smoke','--device','cpu','--epochs','2']))
        self.assertEqual(config['batch_size'],2)
        self.assertEqual(config['registry_name'],'signature-yolox-s-smoke')
        self.assertFalse(config['fp16'])

    def test_invalid_output_name_is_rejected(self):
        with self.assertRaises(ValueError):
            make_config(parser().parse_args(['--output','outputs/bad;command']))


class TriggerTests(unittest.TestCase):
    def test_accepted_trigger_is_not_reported_as_completed_training(self):
        response = mock.MagicMock()
        response.__enter__.return_value = types.SimpleNamespace(status=204,read=lambda:b'')
        opener = mock.Mock(return_value=response)
        result = dispatch('team/project',revision='a'*40,epochs=2,token='private-test-value',opener=opener)
        request=opener.call_args.args[0]
        self.assertNotIn('private-test-value',request.full_url)
        payload=json.loads(request.data)
        self.assertEqual(payload['inputs']['epochs'],'2')
        self.assertFalse(result['training_completion_verified'])

    def test_untrusted_inputs_do_not_reach_network_or_shell(self):
        opener = mock.Mock()
        for kwargs in ({'repo':'../bad','revision':'a'*40},
                       {'repo':'a/b','revision':'main;bad'},
                       {'repo':'a/b','revision':'a'*40,'epochs':30}):
            with self.assertRaises(ValueError):
                dispatch(**kwargs,token='test',opener=opener)
        opener.assert_not_called()


class ApiError(Exception):
    def __init__(self, status):
        self.response={'ResponseMetadata':{'HTTPStatusCode':status}}


class LockTests(unittest.TestCase):
    def store(self):
        client=mock.Mock()
        return types.SimpleNamespace(bucket='bucket',client=client,key=lambda k:'prefix/'+k,
                                     get_json=mock.Mock(),_error=lambda e,op:RuntimeError(op))

    def test_lock_uses_atomic_create_and_second_writer_cannot_enter(self):
        store=self.store();lock=RegistryLock(store,'model','run')
        lock.__enter__()
        self.assertEqual(store.client.put_object.call_args.kwargs['IfNoneMatch'],'*')
        store.client.put_object.side_effect=ApiError(412)
        with self.assertRaises(RuntimeError):
            RegistryLock(store,'model','other').__enter__()

    def test_foreign_lock_is_never_deleted(self):
        store=self.store();lock=RegistryLock(store,'model','run')
        store.get_json.return_value={'owner':'someone-else'}
        with self.assertRaises(RuntimeError):
            lock.__exit__(None,None,None)
        store.client.delete_object.assert_not_called()

    def test_own_lock_is_released(self):
        store=self.store();lock=RegistryLock(store,'model','run')
        store.get_json.return_value={'owner':lock.owner}
        lock.__exit__(None,None,None)
        store.client.delete_object.assert_called_once()


if __name__ == '__main__':
    unittest.main()
