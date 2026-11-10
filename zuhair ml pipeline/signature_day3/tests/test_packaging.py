"""Regression coverage for the extensionless LICENSE omitted in the first package."""
import json
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_day2 import scratch
from export_model import HANDOFF_FILES, HANDOFF_SOURCE_FILES, package_handoff, validate_handoff_sources


class HandoffPackagingTests(unittest.TestCase):
    def test_extensionless_license_survives_smoke_metadata_repack(self):
        with scratch() as temporary:
            root = Path(temporary)
            for name in HANDOFF_FILES:
                (root/name).write_bytes(('fixture-'+name).encode())
            (root/'metadata.json').write_text(json.dumps({'smoke_run':True}),encoding='utf-8')
            package_handoff(root)
            (root/'metadata.json').write_text(json.dumps({'smoke_run':True,'promotion_eligible':False}),encoding='utf-8')
            with zipfile.ZipFile(package_handoff(root)) as bundle:
                self.assertEqual(set(bundle.namelist()),set(HANDOFF_FILES))
                self.assertEqual(bundle.read('LICENSE'),b'fixture-LICENSE')
                self.assertTrue(json.loads(bundle.read('metadata.json'))['smoke_run'])
                self.assertFalse(json.loads(bundle.read('metadata.json'))['promotion_eligible'])

    def test_missing_license_prevents_creation_of_incomplete_handoff(self):
        with scratch() as temporary:
            root = Path(temporary)
            for name in HANDOFF_FILES:
                if name != 'LICENSE':
                    (root/name).write_bytes(b'fixture')
            with self.assertRaisesRegex(FileNotFoundError,'LICENSE'):
                package_handoff(root)
            self.assertFalse((root/'farhan_handoff.zip').exists())

    def test_source_assets_fail_early_when_license_is_missing(self):
        with scratch() as temporary:
            root = Path(temporary)
            for name in HANDOFF_SOURCE_FILES:
                if name != 'LICENSE':
                    (root/name).write_bytes(b'fixture')
            with self.assertRaisesRegex(FileNotFoundError,'LICENSE'):
                validate_handoff_sources(root)
        validate_handoff_sources()


if __name__ == '__main__':
    unittest.main()
