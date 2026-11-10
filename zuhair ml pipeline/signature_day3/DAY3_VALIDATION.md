# Day 3 validation

Built 2026-11-09. No real model metrics or cloud deployment are claimed here.

Offline suite: **61 tests passed**, including the retained 34 Day 2 tests, 24
Day 3 tests and 3 packaging regression tests. Coverage includes strict validation improvement, ties/regressions,
invalid/missing AP, mismatched evaluation protocol/labels, smoke exclusion,
incomplete training/parity/acceptance rejection, one-to-one inference behavior,
bounded subset positives/negatives, default batch settings, safe trigger input
handling, atomic lock acquisition and refusing to delete another writer's lock.

Revision 1 restores the extensionless project LICENSE omitted from revision 0,
checks required source assets before building/training, and preserves LICENSE
when repacking smoke metadata. Regression tests verify the actual handoff ZIP
contents, missing-license failures and updated metadata after a repack.

User-provided Colab logs confirmed Linux bootstrap/model checks, MLflow database
creation and two CPU epochs with verified R2 checkpoints. That run then failed
on the missing LICENSE during handoff packaging; it did not finish upload/gating.

Python source and every notebook code cell are syntax checked. The package builder
verifies ZIP integrity and exact byte equality between source, embedded notebook
payload and packaged source. CLI dry-run/help are checked without installing
training dependencies.

Unverified in this Windows workspace: Linux bootstrap package resolution, live
MLflow 2.22.0/S3 tracking, Cloudflare permissions and conditional locking, actual
YOLOX training, real ONNX export/parity/quality, final Day 2 baseline contents,
GPU availability and a real GitHub workflow dispatch. No cloud credentials were
used, no remote job was launched, and no production pointer was modified.

Run the bounded CPU trial first in the connected environment. A smoke-completed
run can have failing model image acceptance and is never promotion eligible.
Inspect its actual evidence. A full run needs the final Day 2 R2 manifest and
improved validation AP plus passing completion checks before promotion.
