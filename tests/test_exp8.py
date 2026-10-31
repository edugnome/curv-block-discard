"""Exp8 runner wiring: backbone dispatch, seed aggregation, and Phi measurement.

White-box, mostly dataset-free checks that the Stage-A diagnostic runner composes the
parameter-space estimator, the backbone segmentation, and the checkpoint schedule into a
JSON-safe Phi report on a real ResNet-18, and that seed aggregation behaves. The
checkpoint schedule itself (``checkpoint_epochs``, ``epoch_to_label``) is shared
protocol, not exp8's own, and is tested once in `test_protocol.py`. The numerical
fidelity of Phi, the diagonal norms, and the coupling field is covered exactly in
`test_param_space`; here we verify the integration seam.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet50_Weights, resnet50

import experiments.runner_exp8 as runner_exp8  # noqa: E402
from experiments.config import Exp8Config  # noqa: E402
from experiments.data import (  # noqa: E402
    _HFImageClassification,
    _imagenet32_transforms,
)
from experiments.models import (  # noqa: E402
    SegmentedPlainResNet18,
    SegmentedResNet18,
    SegmentedResNet34,
    SegmentedResNet50,
    SegmentedWideResNet18,
)
from experiments.phi import measure_phi  # noqa: E402
from experiments.regroup import Regrouped  # noqa: E402
from experiments.runner_exp8 import (  # noqa: E402
    Exp8Runner,
    _build_model,
    add_args,
    aggregate,
    build_config,
)

_DEVICE = torch.device("cpu")


def _tiny_image_loaders(size: int = 16) -> tuple[DataLoader, DataLoader]:
    """Tiny CIFAR-shaped loaders so a full ``Exp8Runner.run()`` trains for real, fast."""
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(4, 3, size, size), torch.randint(0, 10, (4,)))
    return DataLoader(data, batch_size=2), DataLoader(data, batch_size=2)


def _resnet_batch(
    seed: int = 0,
) -> tuple[SegmentedResNet18, torch.Tensor, torch.Tensor]:
    """A real ResNet-18 (`ParamGroupedModel`) and a CIFAR-shaped classification minibatch."""
    torch.manual_seed(seed)
    model = SegmentedResNet18(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    return model, x, y


def test_build_model_dispatch() -> None:
    cfg = Exp8Config()

    m18 = _build_model(cfg, "resnet18")
    assert isinstance(m18, SegmentedResNet18) and not isinstance(
        m18, (SegmentedPlainResNet18, SegmentedResNet50)
    ), "resnet18 key must build a plain SegmentedResNet18"

    m_plain = _build_model(cfg, "plain_resnet18")
    assert isinstance(m_plain, SegmentedPlainResNet18), (
        "plain_resnet18 key must build SegmentedPlainResNet18"
    )

    # Verify pretrained dispatch without reading caches or fetching ImageNet weights.
    with (
        patch.dict("os.environ", {"DAG_HESSE_ALLOW_DOWNLOADS": "1"}),
        patch(
            "experiments.models.tv_models.resnet50",
            side_effect=lambda **kwargs: resnet50(weights=None),
        ) as build_backbone,
    ):
        m50 = _build_model(Exp8Config.a2_pretrained(), "resnet50")
    build_backbone.assert_called_once_with(weights=ResNet50_Weights.IMAGENET1K_V2)
    assert isinstance(m50, SegmentedResNet50), (
        "resnet50 key must build SegmentedResNet50"
    )

    raised = False
    try:
        _build_model(cfg, "vit")
    except ValueError:
        raised = True
    assert raised, "unknown arch must raise ValueError"


def test_build_model_resnet34_and_width() -> None:
    cfg = Exp8Config.a1_width()

    m34 = _build_model(cfg, "resnet34")
    assert isinstance(m34, SegmentedResNet34), "resnet34 key builds SegmentedResNet34"

    m_half = _build_model(cfg, "resnet18_w0.5")
    assert isinstance(m_half, SegmentedWideResNet18), (
        "width key builds the wide variant"
    )
    half = sum(p.numel() for p in m_half.parameters())
    full = sum(p.numel() for p in _build_model(cfg, "resnet18").parameters())
    assert half < full, "0.5x width must have fewer parameters than 1x"


def test_build_model_pretrained_guard() -> None:
    cfg = Exp8Config(pretrained=True)  # from-scratch arch + pretrained -> rejected
    raised = False
    try:
        _build_model(cfg, "resnet18")
    except ValueError:
        raised = True
    assert raised, "pretrained=True with a from-scratch arch must raise"


def test_measure_phi_structure() -> None:
    model, x, y = _resnet_batch()
    report = measure_phi(model, x, y, nn.CrossEntropyLoss(), n_probes=1)

    assert set(report) == {"phi", "diag_frob", "coupling", "group_sizes"}, "report keys"
    assert isinstance(report["phi"], float) and not math.isinf(report["phi"]), (
        "phi must be a finite float (or NaN)"
    )

    names = list(model.get_param_groups().keys())
    assert list(report["diag_frob"].keys()) == names, "diag_frob must be group-major"
    assert all(isinstance(v, float) for v in report["diag_frob"].values())

    groups = list(model.get_param_groups().values())
    assert report["group_sizes"] == [sum(p.numel() for p in ps) for ps in groups]

    expected_pairs = {f"{v}->{w}" for i, v in enumerate(names) for w in names[i + 1 :]}
    assert set(report["coupling"]) == expected_pairs, (
        "coupling must cover all v<w pairs"
    )
    assert all(isinstance(v, float) for v in report["coupling"].values())


def test_aggregate() -> None:
    raw = {
        "resnet18": {
            "init": {
                "seed_1": {
                    "phi": 0.4,
                    "diag_frob": {"stem": 1.0, "head": 2.0},
                    "coupling": {"stem->head": 0.3},
                    "test_acc": 0.10,
                },
                "seed_2": {
                    "phi": 0.6,
                    "diag_frob": {"stem": 3.0, "head": 4.0},
                    "coupling": {"stem->head": 0.5},
                    "test_acc": 0.20,
                },
            }
        }
    }
    agg = aggregate(raw)["resnet18"]["init"]

    assert agg["seeds"] == raw["resnet18"]["init"], "raw per-seed reports are kept"
    assert abs(agg["phi_mean"] - 0.5) < 1e-9 and abs(agg["phi_std"] - 0.1) < 1e-9
    assert abs(agg["diag_frob"]["stem"]["mean"] - 2.0) < 1e-9
    assert abs(agg["diag_frob"]["head"]["mean"] - 3.0) < 1e-9
    assert abs(agg["coupling"]["stem->head"]["mean"] - 0.4) < 1e-9
    assert abs(agg["test_acc_mean"] - 0.15) < 1e-9
    assert agg["n_seeds"] == 2


def test_a1_imagenet32_profile() -> None:
    cfg = Exp8Config.a1_imagenet32()
    assert cfg.dataset == "imagenet32", "a1_imagenet32 measures on ImageNet-32"
    assert cfg.num_classes == 1001, "the HF mirror keeps the index-0 background class"
    assert cfg.archs == ["resnet18", "plain_resnet18"], "same inversion pair as a1"


def test_save_checkpoints_writes_files_and_records_their_paths_per_seed(
    tmp_path: Path,
) -> None:
    with patch.object(
        runner_exp8,
        "get_cifar100_image_loaders",
        lambda *_a, **_k: _tiny_image_loaders(),
    ):
        cfg = Exp8Config(
            archs=["resnet18"],
            num_classes=10,
            n_probes=1,
            hessian_batch_size=2,
            checkpoint_epochs=["init", "final"],
            training=replace(
                Exp8Config().training,
                epochs=1,
                seeds=[7],
                batch_size=2,
                deterministic=False,
            ),
            output_dir=str(tmp_path),
            save_checkpoints=True,
        )
        bundle = Exp8Runner(cfg, _DEVICE).run()

    init_seed_report = bundle["resnet18"]["init"]["seeds"]["seed_7"]
    want_paths = {
        "0": "checkpoints/resnet18_seed7_ep0.pt",
        "1": "checkpoints/resnet18_seed7_ep1.pt",
    }
    assert init_seed_report["checkpoints"] == want_paths
    for rel in want_paths.values():
        assert (tmp_path / rel).is_file(), f"{rel} was not written"
    # Every checkpoint of a seed shares the same (whole-run) path map.
    assert bundle["resnet18"]["final"]["seeds"]["seed_7"]["checkpoints"] == want_paths


def _train_and_save(tmp_path: Path) -> tuple[Exp8Config, dict]:
    """A tiny two-epoch save_checkpoints run under ``tmp_path``; returns the config used
    (so a caller can `replace` it into a restore config pointed at the same directory)
    together with the bundle that one run already produced."""
    cfg = Exp8Config(
        archs=["resnet18"],
        num_classes=10,
        n_probes=2,
        hessian_batch_size=2,
        checkpoint_epochs=["init", "final"],
        training=replace(
            Exp8Config().training,
            epochs=2,
            seeds=[7],
            batch_size=2,
            deterministic=False,
        ),
        output_dir=str(tmp_path),
        save_checkpoints=True,
    )
    return cfg, Exp8Runner(cfg, _DEVICE).run()


def test_restore_checkpoints_dir_never_calls_trainer_train(tmp_path: Path) -> None:
    """Constraint: no optimisation step runs at all when restoring - proven directly,
    not inferred, by making `Trainer.train` raise if the restore path ever reaches it.
    """
    with patch.object(
        runner_exp8,
        "get_cifar100_image_loaders",
        lambda *_a, **_k: _tiny_image_loaders(),
    ):
        base, _ = _train_and_save(tmp_path)

        restore_cfg = replace(
            base,
            save_checkpoints=False,
            restore_checkpoints_dir=str(tmp_path),
            output_dir=str(tmp_path / "restored"),
        )
        with patch.object(
            runner_exp8.Trainer,
            "train",
            side_effect=AssertionError(
                "Trainer.train must not be called when restore_checkpoints_dir is set"
            ),
        ):
            Exp8Runner(restore_cfg, _DEVICE).run()


def test_restore_checkpoints_dir_builds_no_train_loader(tmp_path: Path) -> None:
    """Restoring must not even build a train loader - wasted dataset work, and the
    surface `test_exp8_rebuilds_the_train_loader_once_per_seed` guards would otherwise
    also apply here: one call for the shared val loader, none per seed.
    """
    calls: list[tuple[DataLoader, DataLoader]] = []

    def fake_loaders(*_a: object, **_k: object) -> tuple[DataLoader, DataLoader]:
        pair = _tiny_image_loaders()
        calls.append(pair)
        return pair

    with patch.object(runner_exp8, "get_cifar100_image_loaders", fake_loaders):
        base, _ = _train_and_save(tmp_path)

        calls.clear()
        restore_cfg = replace(
            base,
            save_checkpoints=False,
            restore_checkpoints_dir=str(tmp_path),
            output_dir=str(tmp_path / "restored"),
        )
        Exp8Runner(restore_cfg, _DEVICE).run()

    assert len(calls) == 1, "restoring must only build the shared val loader"


def test_restore_checkpoints_dir_reproduces_the_training_paths_measurement(
    tmp_path: Path,
) -> None:
    """The round trip: train briefly with save_checkpoints, then - in a *separate*
    Exp8Runner.run() call, with no training at all - restore from the written files
    and measure again, at both a from-scratch ("init") and an actually-trained
    ("final") checkpoint.

    `measure_phi`'s headline call seeds its Hutchinson probes off the ambient global
    RNG, not off anything tied to the checkpoint's weights (see its own docstring):
    training between "init" and "final" advances that ambient state by a different
    amount than restoring the same two checkpoints does (no data loader is even
    built), so `phi` / `diag_frob` / `coupling` are not expected to come out
    bit-identical across these two separate runs - only the same *shape*, which is
    constraint 2 ("a bundle produced from checkpoints is the same shape as one
    produced by training"). What must match exactly, because neither depends on any
    RNG at all, is what a restore actually has to get right: `test_acc`, computed
    from the weights alone on the fixed validation batch - the fingerprint constraint
    5 asks for - and the architecture-only `group_sizes`.
    """
    with patch.object(
        runner_exp8,
        "get_cifar100_image_loaders",
        lambda *_a, **_k: _tiny_image_loaders(),
    ):
        base, trained_bundle = _train_and_save(tmp_path)

        restore_cfg = replace(
            base,
            save_checkpoints=False,
            restore_checkpoints_dir=str(tmp_path),
            output_dir=str(tmp_path / "restored"),
        )
        restored_bundle = Exp8Runner(restore_cfg, _DEVICE).run()

    for ckpt in ("init", "final"):
        trained_report = trained_bundle["resnet18"][ckpt]["seeds"]["seed_7"]
        restored_report = restored_bundle["resnet18"][ckpt]["seeds"]["seed_7"]

        assert restored_report["test_acc"] == trained_report["test_acc"], ckpt
        assert restored_report["group_sizes"] == trained_report["group_sizes"], ckpt
        assert set(restored_report["diag_frob"]) == set(trained_report["diag_frob"]), (
            ckpt
        )
        assert set(restored_report["coupling"]) == set(trained_report["coupling"]), ckpt
        assert math.isfinite(restored_report["phi"]), ckpt
        assert "checkpoints" not in restored_report, (
            "restore_cfg did not ask to re-save"
        )


def test_restore_checkpoints_dir_cli_override() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    default = build_config(parser.parse_args([]))
    assert default.restore_checkpoints_dir is None

    cfg = build_config(
        parser.parse_args("--restore-checkpoints-dir results/exp8_a1_long_s42".split())
    )
    assert cfg.restore_checkpoints_dir == "results/exp8_a1_long_s42"
    assert cfg.save_checkpoints is False, "restoring does not imply re-saving"


def test_exp8_rebuilds_the_train_loader_once_per_seed() -> None:
    """Check loader factory call count and object identity per architecture and seed.

    Separate training loaders isolate persistent-worker RNG state.
    """
    calls: list[tuple[DataLoader, DataLoader]] = []

    def fake_loaders(*_a: object, **_k: object) -> tuple[DataLoader, DataLoader]:
        pair = _tiny_image_loaders()
        calls.append(pair)
        return pair

    with patch.object(runner_exp8, "get_cifar100_image_loaders", fake_loaders):
        cfg = Exp8Config(
            archs=["resnet18"],
            num_classes=10,
            n_probes=1,
            hessian_batch_size=2,
            checkpoint_epochs=["init"],
            training=replace(
                Exp8Config().training, epochs=0, seeds=[7, 8], batch_size=2
            ),
        )
        Exp8Runner(cfg, _DEVICE).run()

    assert len(calls) == 3, (
        f"one call for the shared val loader, plus one per seed - got {len(calls)}"
    )
    seed_7_loader, seed_8_loader = calls[1][0], calls[2][0]
    assert seed_7_loader is not seed_8_loader, (
        "seed 7 and seed 8 must not share the same train DataLoader object"
    )


def test_without_save_checkpoints_no_checkpoints_key_appears() -> None:
    with patch.object(
        runner_exp8,
        "get_cifar100_image_loaders",
        lambda *_a, **_k: _tiny_image_loaders(),
    ):
        cfg = Exp8Config(
            archs=["resnet18"],
            num_classes=10,
            n_probes=1,
            hessian_batch_size=2,
            checkpoint_epochs=["init"],
            training=replace(Exp8Config().training, epochs=0, seeds=[7], batch_size=2),
        )
        bundle = Exp8Runner(cfg, _DEVICE).run()
    assert "checkpoints" not in bundle["resnet18"]["init"]["seeds"]["seed_7"]


def _run_a2_style(
    probe_sweep: list[int] | None = None,
    n_probes: int = 1,
    with_rank1: bool = False,
    granularity: list[str] | None = None,
) -> dict:
    """A single-checkpoint (epochs=0) exp8 run, real ResNet-18, tiny stubbed data."""
    with patch.object(
        runner_exp8,
        "get_cifar100_image_loaders",
        lambda *_a, **_k: _tiny_image_loaders(),
    ):
        cfg = Exp8Config(
            archs=["resnet18"],
            num_classes=10,
            n_probes=n_probes,
            hessian_batch_size=2,
            checkpoint_epochs=["init"],
            training=replace(Exp8Config().training, epochs=0, seeds=[7], batch_size=2),
            probe_sweep=probe_sweep or [],
            with_rank1=with_rank1,
            granularity=granularity or [],
        )
        return Exp8Runner(cfg, _DEVICE).run()


def test_without_probe_sweep_no_phi_by_probes_key_appears() -> None:
    bundle = _run_a2_style([])
    assert "phi_by_probes" not in bundle["resnet18"]["init"]["seeds"]["seed_7"]


def test_probe_sweep_is_isolated_from_the_headline_measurement() -> None:
    seed_report = _run_a2_style([2, 5], n_probes=1)["resnet18"]["init"]["seeds"][
        "seed_7"
    ]
    phi_by_probes = seed_report["phi_by_probes"]

    assert set(phi_by_probes) == {"2", "5"}
    assert all(isinstance(v, float) for v in phi_by_probes.values())

    # The headline measurement (n_probes) draws from the ambient global RNG before the
    # sweep runs; changing how much it drew must not move the sweep's own estimates,
    # since each of its calls is seeded off the run's own seed, not off the RNG's
    # position when the sweep starts.
    other_headline = _run_a2_style([2, 5], n_probes=7)["resnet18"]["init"]["seeds"][
        "seed_7"
    ]["phi_by_probes"]
    assert other_headline == phi_by_probes


def test_without_with_rank1_no_rank1_keys_appear() -> None:
    seed_report = _run_a2_style()["resnet18"]["init"]["seeds"]["seed_7"]
    assert "phi_rank1" not in seed_report and "lambda_max" not in seed_report


def test_with_rank1_adds_phi_rank1_and_lambda_max() -> None:
    seed_report = _run_a2_style(with_rank1=True)["resnet18"]["init"]["seeds"]["seed_7"]
    assert isinstance(seed_report["phi_rank1"], float)
    assert isinstance(seed_report["lambda_max"], float)


def test_without_granularity_no_granularity_keys_appear() -> None:
    seed_report = _run_a2_style()["resnet18"]["init"]["seeds"]["seed_7"]
    assert "phi_by_granularity" not in seed_report
    assert "diag_frob_by_granularity" not in seed_report


def test_granularity_skips_stage_and_measures_through_regrouped() -> None:
    seed_report = _run_a2_style(granularity=["stage", "block"])["resnet18"]["init"][
        "seeds"
    ]["seed_7"]
    by_level = seed_report["phi_by_granularity"]
    diag_by_level = seed_report["diag_frob_by_granularity"]

    assert set(by_level) == {"block"}, "stage duplicates the headline partition"
    assert isinstance(by_level["block"], float), "phi_by_granularity stays a bare float"

    assert set(diag_by_level) == {"block"}, "stage duplicates the headline partition"
    expected_groups = list(
        Regrouped(SegmentedResNet18(num_classes=10), "block").get_param_groups()
    )
    assert list(diag_by_level["block"]) == expected_groups, "group-major order"
    assert all(isinstance(v, float) for v in diag_by_level["block"].values()), (
        "norms alone, no coupling field"
    )


def test_hf_image_classification_wrapper() -> None:
    split = [{"image": Image.new("RGB", (32, 32)), "label": 7} for _ in range(3)]
    ds = _HFImageClassification(split, _imagenet32_transforms(train=False))
    assert len(ds) == 3, "wrapper length follows the split"
    image, label = ds[0]
    assert image.shape == (3, 32, 32), "32x32 RGB tensor"
    assert image.dtype == torch.float32 and label == 7, "normalized tensor + int label"
