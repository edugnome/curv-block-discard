"""CPU scalar/stub contract tests. Never import the model/HVP/data stack."""

import copy
import hashlib
import importlib
import json
import math
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def modules():
    return tuple(
        importlib.import_module("experiments.crossing." + n)
        for n in ("verify", "analysis", "measure")
    )


def synthetic_manifest():
    v, _, _ = modules()
    rows = [
        {
            "arch": arch,
            "seed": seed,
            "epoch": 300,
            "path": f"/synthetic/s{seed}-{arch}.pt",
            "sha256": "0" * 64,
        }
        for seed in (42, 43, 44)
        for arch in ("resnet18", "plain_resnet18")
    ]
    m = v.manifest_contract("/synthetic/data", rows)
    m.update(
        source_hashes=v.scientific_source_hashes(),
        data_hashes={
            f"/synthetic/data/cifar-100-python/{name}": "0" * 64
            for name in ("test", "meta")
        },
        packages={
            "torch": "synthetic",
            "torchvision": "synthetic",
            "numpy": "synthetic",
        },
        provenance_limits="synthetic test only",
    )
    return m


def components(scale=1.0, layer=False):
    d = [scale * (0.7 if layer else 1.0), scale * 2.0]
    return {"diag_sq": d, "full_sq": scale * (8.0 if layer else 6.0)}


def synthetic_cells(v):
    return [
        {
            "id": c["id"],
            "status": "complete",
            "pairs": [
                {
                    "replicate": i,
                    "stage": {
                        "diag_sq": [1 + i / 70, 2 + (i % 3) / 20],
                        "full_sq": 6 + (i % 7) / 13,
                    },
                    "layer": {
                        "diag_sq": [0.9 + i / 50, 2.1 + (i % 7) / 30],
                        "full_sq": 9 + (i % 5) / 4,
                    },
                }
                for i in range(30)
            ],
        }
        for c in v.roster()
    ]


class ManifestTests(unittest.TestCase):
    v: ModuleType
    a: ModuleType
    m: ModuleType
    manifest: dict[str, Any]

    @classmethod
    def setUpClass(cls):
        cls.v, cls.a, cls.m = modules()
        cls.manifest = synthetic_manifest()

    def test_frozen_manifest_and_roster(self):
        self.v.validate_manifest(self.manifest)
        self.assertEqual(len(self.v.roster()), 12)
        self.assertEqual(len(self.manifest["streams"]), 720)
        self.assertEqual(len({s["seed"] for s in self.manifest["streams"]}), 720)
        self.assertEqual(len({s["id"] for s in self.manifest["streams"]}), 720)
        self.assertEqual(len(self.manifest["checkpoints"]), 6)
        self.assertEqual(self.manifest["statistics"], self.v.STATISTICS)

    def test_reject_stream_mutation(self):
        for key, value in (("seed", 7), ("id", "other")):
            m = copy.deepcopy(self.manifest)
            m["streams"][0][key] = value
            with self.assertRaises(ValueError):
                self.v.validate_manifest(m)

    def test_reject_cohort_and_budget_mutation(self):
        for key, value in (("replicates", 29), ("probes", 101), ("cells", [])):
            m = copy.deepcopy(self.manifest)
            m[key] = value
            with self.assertRaises(ValueError):
                self.v.validate_manifest(m)

    def test_input_disjoint_ordered_and_no_download(self):
        self.assertEqual(self.manifest["batches"], self.v.BATCHES)
        m = copy.deepcopy(self.manifest)
        m["batches"][1]["indices"][0] = 0
        with self.assertRaises(ValueError):
            self.v.validate_manifest(m)
        for key in ("train", "download"):
            m = copy.deepcopy(self.manifest)
            m["dataset"][key] = True
            with self.assertRaises(ValueError):
                self.v.validate_manifest(m)

    def test_hash_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source"
            p.write_text("one")
            ledger = {str(p): hashlib.sha256(b"one").hexdigest()}
            self.assertEqual(self.v.check_hashes(ledger), [])
            p.write_text("two")
            self.assertTrue(self.v.check_hashes(ledger))
            p.unlink()
            self.assertTrue(self.v.check_hashes(ledger))

    def test_cli_interfaces_and_no_unreviewed_launch(self):
        self.m.parser().parse_args(
            ["--manifest", "a", "--output", "b", "--execute-reviewed"]
        )
        self.a.parser().parse_args(
            ["--manifest", "a", "--run-dir", "b", "--output", "c"]
        )
        self.v.parser().parse_args(["--manifest", "a", "--metadata-only"])
        with self.assertRaises(ValueError):
            self.m.launch_guard(False, "0")
        for device in (None, "", "0,1"):
            with self.assertRaises(ValueError):
                self.m.launch_guard(True, device)

    def test_batch_uses_test_split_without_download(self):
        import torch

        calls = []

        class Dataset:
            def __init__(self, root, *, train, download, transform):
                calls.append((root, train, download))

            def __getitem__(self, index):
                return torch.zeros(3, 32, 32), index

        class Vision:
            class datasets:
                CIFAR100 = Dataset

        cell = self.v.roster()[0]
        x, y, indices = self.m.make_test_batch(self.manifest, cell, torch, Vision)
        self.assertEqual(calls, [(self.manifest["dataset"]["root"], False, False)])
        self.assertEqual(list(x.shape), [32, 3, 32, 32])
        self.assertEqual(y.tolist(), indices)


class ScalarTests(unittest.TestCase):
    v: ModuleType
    a: ModuleType
    m: ModuleType
    manifest: dict[str, Any]

    @classmethod
    def setUpClass(cls):
        cls.v, cls.a, cls.m = modules()

    def test_pool_before_nonlinearity(self):
        x = [
            {"diag_sq": [1.0, 9.0], "full_sq": 4.0},
            {"diag_sq": [9.0, 1.0], "full_sq": 16.0},
        ]
        p = self.a.pool(x)
        self.assertAlmostEqual(p["S"], 2.0)
        self.assertNotAlmostEqual(p["S"], (4.0 + 1.0) / 2)
        self.assertAlmostEqual(p["Phi"], 0.0)
        self.assertAlmostEqual(p["Phi_sep"], 0.5)

    def test_no_clipping(self):
        self.assertLess(
            self.a.pool([{"diag_sq": [4.0, 9.0], "full_sq": 2.0}])["Phi"], 0
        )
        self.assertGreater(
            self.a.pool([{"diag_sq": [4.0, 9.0], "full_sq": 2.0}])["S"], 1
        )

    def test_domains(self):
        for d, f in (
            ([1.0, -1.0], 1.0),
            ([math.nan, 1.0], 1.0),
            ([1.0, 1.0], 0.0),
            ([1.0, 1.0], math.inf),
            ([0.0, 1.0], 3.0),
        ):
            with self.assertRaises(ValueError):
                self.a.pool([{"diag_sq": d, "full_sq": f}])

    def test_reconstruction_and_order(self):
        p = self.m.reconstruct(
            {"phi": -0.5, "diag_frob": {"a": 2.0, "b": 3.0}}, ["a", "b"]
        )
        self.assertEqual(p["diag_sq"], [4.0, 9.0])
        self.assertAlmostEqual(p["full_sq"], 13 / 1.5)
        for phi in (1.0, math.nan, 1.01):
            with self.assertRaises(ValueError):
                self.m.reconstruct(
                    {"phi": phi, "diag_frob": {"a": 2.0, "b": 3.0}}, ["a", "b"]
                )
        with self.assertRaises(ValueError):
            self.m.reconstruct(
                {"phi": 0.1, "diag_frob": {"b": 1.0, "a": 1.0}}, ["a", "b"]
            )

    def test_partition_coverage_and_overlap(self):
        class P:
            requires_grad = True

            def numel(self):
                return 1

        p, q = P(), P()
        expected = [
            {"name": "a", "size": 1, "parameters": ["p"]},
            {"name": "b", "size": 1, "parameters": ["q"]},
        ]
        self.assertEqual(
            self.m.partition_contract({"a": [p], "b": [q]}, {"p": p, "q": q}, expected),
            expected,
        )
        for groups in ({"a": [p], "b": [p]}, {"a": [p], "b": []}):
            with self.assertRaises(ValueError):
                self.m.partition_contract(groups, {"p": p, "q": q}, expected)

    def test_linear_quantiles_and_ddof(self):
        self.assertAlmostEqual(self.a.quantile([0.0, 10.0], 0.025), 0.25)
        self.assertAlmostEqual(self.a.quantile([0.0, 10.0], 0.95), 9.5)
        self.assertAlmostEqual(self.a.sample_sd([1.0, 3.0]), math.sqrt(2))

    def test_joint_resampling_and_independent_cells(self):
        import numpy as np

        synthetic_cells(self.v)
        rng = np.random.default_rng(20261020)
        indices = self.a.resample_indices(rng, 2, 100)
        self.assertFalse(np.array_equal(indices[0], indices[1]))
        stage = [components(1 + i / 7) for i in range(30)]
        layer = [
            {
                "diag_sq": [2 * x["diag_sq"][0], 2 * x["diag_sq"][1]],
                "full_sq": 2 * x["full_sq"],
            }
            for x in stage
        ]
        b = self.a.bootstrap_cell(stage, layer, indices[0])
        self.assertTrue(np.allclose(b["stage"][:, 0], b["layer"][:, 0]))
        self.assertTrue(np.allclose(b["delta"], 0))

    def test_invalid_bootstrap_draw_is_not_dropped(self):
        import numpy as np

        stage = [{"diag_sq": [0.0, 1.0], "full_sq": 3.0} for _ in range(29)] + [
            {"diag_sq": [1.0, 1.0], "full_sq": 3.0}
        ]
        self.a.pool(stage)  # Original pooled point is regular.
        indices = np.zeros((2, 30), dtype=int)
        with self.assertRaises(ValueError):
            self.a.bootstrap_cell(stage, stage, indices)

    def test_fixed_centered_max_screen(self):
        import numpy as np

        observed = np.arange(36, dtype=float)
        boot = np.array([observed - 1, observed, observed + 2, observed + 3])
        screen = self.a.max_screen(observed, boot)
        sd = boot.std(axis=0, ddof=1)
        expected = self.a.quantile(np.max(np.abs(boot - observed) / sd, axis=1), 0.95)
        self.assertAlmostEqual(screen["critical_value"], expected)
        self.assertEqual(len(screen["intervals"]), 36)
        with self.assertRaises(ValueError):
            self.a.max_screen(observed[:35], boot[:, :35])

    def test_crossing_requires_opposite_exclusions(self):
        self.assertEqual(self.a.classify([1.01, 1.1], [0.7, 0.99]), "resolved_crossing")
        self.assertEqual(self.a.classify([1.01, 1.1], [0.7, 1.01]), "unresolved")
        self.assertEqual(
            self.a.classify([0.8, 0.99], [1.01, 1.2]), "resolved_reverse_crossing"
        )

    def test_jackknife_diagnostic(self):
        x = [{"diag_sq": [1.0, 1.0], "full_sq": f} for f in (2.0, 3.0, 4.0)]
        j = self.a.jackknife(x)
        loo = [self.a.pool(x[:i] + x[i + 1 :])["S"] for i in range(3)]
        mean = sum(loo) / 3
        self.assertAlmostEqual(j["S"]["bias"], 2 * (mean - self.a.pool(x)["S"]))
        self.assertAlmostEqual(
            j["S"]["se"], math.sqrt(2 / 3 * sum((z - mean) ** 2 for z in loo))
        )

    def test_full_bootstrap_fixed_family_and_reproducible(self):
        cells = synthetic_cells(self.v)
        result = self.a.analyze(cells)
        repeat = self.a.analyze(cells)
        self.assertEqual(result, repeat)
        self.assertEqual(len(result["cells"]), 12)
        self.assertEqual(len(result["family"]["metrics"]), 36)
        self.assertEqual(result["family"]["status"], "available")
        self.assertEqual(result["statistics"]["resamples"], 10000)
        self.assertEqual(len(result["input_comparisons"]), 6)

    def test_missing_invalid_and_zero_sd_invalidate_whole_family(self):
        for mutate in (
            lambda c: c[0]["pairs"].pop(),
            lambda c: c[0]["pairs"][0]["stage"].update(full_sq=-1),
            lambda c: [p["stage"].update(diag_sq=[0.0, 1.0]) for p in c[0]["pairs"]],
            lambda c: [
                p.update(stage=components(), layer=components(layer=True))
                for p in c[0]["pairs"]
            ],
        ):
            cells = synthetic_cells(self.v)
            mutate(cells)
            r = self.a.analyze(cells)
            self.assertEqual(len(r["cells"]), 12)
            self.assertEqual(r["family"]["status"], "unavailable")
            self.assertTrue(r["family"]["reasons"])
            self.assertEqual(len(r["family"]["metrics"]), 36)


class RecordTests(unittest.TestCase):
    v: ModuleType
    a: ModuleType
    m: ModuleType
    manifest: dict[str, Any]

    @classmethod
    def setUpClass(cls):
        cls.v, cls.a, cls.m = modules()
        cls.manifest = synthetic_manifest()

    def stub_run(self, root):
        import struct

        v, m = self.v, self.manifest
        cell = v.roster()[0]
        d = root / cell["id"]
        d.mkdir()
        (root / "manifest.json").write_text(json.dumps(m))
        mh = v.digest(root / "manifest.json")
        v.write_json(
            root / "run.json",
            {
                "manifest_sha256": mh,
                "cells": [dict(c, status="not_started") for c in v.roster()],
                "caps": v.CAPS,
            },
        )
        cp = next(
            c
            for c in m["checkpoints"]
            if c["arch"] == cell["model"] and c["seed"] == cell["seed"]
        )
        tensors = sorted(
            [
                dict(t, sha256="a" * 64)
                for t in m["partitions"][cell["model"]]["tensors"]
            ],
            key=lambda x: x["name"],
        )
        targets = list(range(32))
        inputs = [
            {
                "name": "X",
                "shape": [32, 3, 32, 32],
                "dtype": "torch.float32",
                "kind": "input",
                "sha256": "b" * 64,
            },
            {
                "name": "y",
                "shape": [32],
                "dtype": "torch.int64",
                "kind": "target",
                "sha256": hashlib.sha256(struct.pack("<32q", *targets)).hexdigest(),
            },
        ]
        identity = {
            "cell": cell,
            "checkpoint": cp,
            "loaded_tensors": tensors,
            "model_tensors": tensors,
            "model_digest": v.object_digest(tensors),
            "input_tensors": inputs,
            "input_digest": v.object_digest(inputs),
            "indices": list(range(32)),
            "ordered_targets": targets,
            "groups": {
                p: m["partitions"][cell["model"]][p] for p in ("stage", "layer")
            },
            "source_hashes": m["source_hashes"],
            "data_hashes": m["data_hashes"],
            "environment": v.ENVIRONMENT,
            "runtime": {
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "autocast": False,
                "floating_dtype": "torch.float32",
            },
        }
        v.write_json(d / "cell-start.json", identity)
        for i in range(30):
            pair = {"cell": cell["id"], "replicate": i, "component_hashes": {}}
            for part in ("stage", "layer"):
                groups = m["partitions"][cell["model"]][part]
                stream = next(
                    s
                    for s in m["streams"]
                    if s["cell"] == cell["id"]
                    and s["replicate"] == i
                    and s["partition"] == part
                )
                o = self.m.reconstruct(
                    {"phi": 0.5, "diag_frob": {g["name"]: 1.0 for g in groups}},
                    [g["name"] for g in groups],
                )
                o.update(
                    cell=cell["id"],
                    replicate=i,
                    partition=part,
                    stream=stream,
                    n_probes=100,
                    groups=groups,
                    state_before=identity["model_digest"],
                    state_after=identity["model_digest"],
                    input_before=identity["input_digest"],
                    input_after=identity["input_digest"],
                    generator_pre="aa",
                    generator_post="bb",
                    adaptation=self.m.ADAPTATION,
                    wall_seconds=0.1,
                )
                v.write_json(d / f"{part}-{i:02d}.json", o)
                pair[part] = o
                pair["component_hashes"][part] = v.digest(d / f"{part}-{i:02d}.json")
            v.write_json(d / f"pair-{i:02d}.json", pair)
        v.write_json(
            d / "cell-complete.json",
            {
                "cell": cell,
                "complete_pairs": 30,
                "model_tensors": tensors,
                "input_tensors": inputs,
                "preservation": {"status": "PASS"},
            },
        )
        rows = [
            dict(
                c,
                status="complete" if i == 0 else "not_started",
                complete_pairs=30 if i == 0 else 0,
            )
            for i, c in enumerate(v.roster())
        ]
        v.write_json(
            root / "campaign.json",
            {
                "manifest_sha256": mh,
                "cells": rows,
                "caps": v.CAPS,
                "complete": False,
                "preservation": {"status": "PASS"},
            },
        )
        v.write_json(root / "preservation.json", {"status": "PASS"})
        return mh, d, identity

    def test_record_roundtrip_and_incomplete_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mh, _, _ = self.stub_run(root)
            cells, _ = self.v.read_run(self.manifest, root, mh)
            self.assertEqual(cells[0]["status"], "complete")
            self.assertEqual(len(cells[0]["pairs"]), 30)
            self.assertFalse(cells[0]["integrity_errors"])
            self.assertEqual(self.a.analyze(cells)["family"]["status"], "unavailable")

    def test_tampered_input_source_runtime_and_cell_identity_rejected(self):
        for key, value in (
            ("cell", {}),
            ("source_hashes", {}),
            ("data_hashes", {}),
            ("ordered_targets", [99] * 32),
            ("environment", {}),
            ("runtime", {}),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                mh, path, identity = self.stub_run(root)
                identity[key] = value
                (path / "cell-start.json").write_text(json.dumps(identity))
                cells, _ = self.v.read_run(self.manifest, root, mh)
                self.assertEqual(cells[0]["status"], "invalid")

    def test_pair_hash_and_missing_observation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mh, path, _ = self.stub_run(root)
            (path / "stage-00.json").unlink()
            cells, _ = self.v.read_run(self.manifest, root, mh)
            self.assertEqual(cells[0]["status"], "invalid")

    def test_failed_preservation_invalidates_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mh, path, _ = self.stub_run(root)
            (root / "preservation.json").write_text(json.dumps({"status": "FAIL"}))
            cells, _ = self.v.read_run(self.manifest, root, mh)
            self.assertTrue(any(c["integrity_errors"] for c in cells))

    def test_tensor_dtype_contract_allows_integer_buffers_not_integer_parameters(self):
        expected = self.v.architecture_contract("resnet18")["tensors"]
        self.assertTrue(
            any(t["dtype"] == "torch.int64" and t["kind"] == "buffer" for t in expected)
        )
        fake = [dict(t, sha256="a" * 64) for t in expected]
        self.m.assert_tensor_schema(fake, expected)
        fake[0]["dtype"] = "torch.float64"
        with self.assertRaises(ValueError):
            self.m.assert_tensor_schema(fake, expected)

    def test_metadata_has_no_heavy_imports(self):
        import subprocess

        command = "import sys; import experiments.crossing.verify; assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"
        subprocess.run([sys.executable, "-I", "-c", command], check=True)


class RuntimeTests(unittest.TestCase):
    v: ModuleType
    a: ModuleType
    m: ModuleType

    @classmethod
    def setUpClass(cls):
        cls.v, cls.a, cls.m = modules()

    def temp(self):
        return tempfile.TemporaryDirectory()

    def test_exclusive_atomic_output(self):
        with self.temp() as d:
            p = Path(d) / "record.json"
            self.v.write_json(p, {"one": 1})
            with self.assertRaises(FileExistsError):
                self.v.write_json(p, {"two": 2})
            self.assertEqual(json.loads(p.read_text()), {"one": 1})
            self.m.create_output(Path(d) / "run")
            with self.assertRaises(FileExistsError):
                self.m.create_output(Path(d) / "run")

    def test_event_timeout_without_sleep(self):
        # Child blocks on a pipe read; communicate uses selector events, not a polling loop.
        result = self.m.capped_subprocess(
            [sys.executable, "-I", "-c", "import os; r,w=os.pipe(); os.read(r, 1)"],
            0.05,
            grace=0.2,
        )
        self.assertTrue(result["timed_out"])
        self.assertLess(result["elapsed_seconds"], 2)
        self.assertNotEqual(result["returncode"], 0)

    def test_timeout_escalates_bounded_termination(self):
        # A fake process avoids a race while deterministically exercising SIGKILL.
        import subprocess

        class P:
            pid = 123
            returncode = -9

            def __init__(self):
                self.calls = 0

            def communicate(self, timeout):
                self.calls += 1
                if self.calls < 3:
                    raise subprocess.TimeoutExpired("stub", timeout)
                return ("captured", None)

        p = P()
        signals = []
        r = self.m.communicate_capped(p, 0.1, 0.1, lambda pid, sig: signals.append(sig))
        self.assertTrue(r["timed_out"])
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(r["stdout"], "captured")

    def test_stop_preserves_orphan_and_full_roster(self):
        with self.temp() as d:
            root = Path(d)

            def worker(cell, path, timeout):
                self.v.write_json(path / "stage-00.json", {"orphan": True})
                return {
                    "timed_out": True,
                    "returncode": -15,
                    "elapsed_seconds": 0.01,
                    "stdout": "stub",
                }

            result = self.m.supervise(
                root, "manifest-hash", worker, self.v.roster(), 100, 10
            )
            self.assertEqual(len(result["cells"]), 12)
            self.assertEqual(result["cells"][0]["status"], "timed_out")
            self.assertTrue(
                all(c["status"] == "not_started" for c in result["cells"][1:])
            )
            self.assertTrue(
                (root / self.v.roster()[0]["id"] / "stage-00.json").exists()
            )
            self.assertFalse(result["complete"])

    def test_campaign_cap_stops_before_cell(self):
        with self.temp() as d:
            values = iter([0.0, 2.0, 2.0, 2.0])

            def worker(*args):
                self.fail("must not start a cell after campaign deadline")

            r = self.m.supervise(
                Path(d),
                "hash",
                worker,
                self.v.roster(),
                1,
                1,
                clock=lambda: next(values),
            )
            self.assertTrue(all(c["status"] == "not_started" for c in r["cells"]))
            self.assertEqual(r["stop_reason"], "campaign_cap")

    def test_final_preservation_failure_cannot_be_complete(self):
        with self.temp() as d:

            def worker(cell, path, timeout):
                for i in range(30):
                    self.v.write_json(path / f"pair-{i:02d}.json", {"stub": True})
                self.v.write_json(path / "cell-complete.json", {"stub": True})
                return {
                    "timed_out": False,
                    "returncode": 0,
                    "stdout": "stub",
                    "elapsed_seconds": 0.0,
                }

            r = self.m.supervise(
                Path(d),
                "hash",
                worker,
                self.v.roster(),
                100,
                10,
                finalizer=lambda remaining: {"status": "FAIL"},
            )
            self.assertFalse(r["complete"])
            self.assertEqual(r["stop_reason"], "preservation_unavailable_or_failed")

    def test_successful_stub_campaign_keeps_all_twelve_cells(self):
        with self.temp() as d:

            def worker(cell, path, timeout):
                for i in range(30):
                    self.v.write_json(path / f"pair-{i:02d}.json", {"stub": True})
                self.v.write_json(path / "cell-complete.json", {"stub": True})
                return {
                    "timed_out": False,
                    "returncode": 0,
                    "stdout": "stub",
                    "elapsed_seconds": 0.0,
                }

            r = self.m.supervise(
                Path(d),
                "hash",
                worker,
                self.v.roster(),
                100,
                10,
                finalizer=lambda remaining: {"status": "PASS"},
            )
            self.assertTrue(r["complete"])
            self.assertEqual(len(r["cells"]), 12)
            self.assertTrue(all(c["complete_pairs"] == 30 for c in r["cells"]))

    def test_failure_stops_without_retry(self):
        with self.temp() as d:
            calls = []

            def worker(cell, path, timeout):
                calls.append(cell["id"])
                return {
                    "timed_out": False,
                    "returncode": 1,
                    "elapsed_seconds": 0,
                    "stdout": "surfaced failure",
                }

            r = self.m.supervise(Path(d), "hash", worker, self.v.roster(), 100, 10)
            self.assertEqual(len(calls), 1)
            self.assertEqual(r["cells"][0]["status"], "failed")

    def test_internal_retry_is_unavailable_not_zero(self):
        self.assertEqual(self.m.ADAPTATION["internal_retry_count"], "UNAVAILABLE")
        self.assertFalse(self.m.ADAPTATION["exact_draw_replay_claimed"])
