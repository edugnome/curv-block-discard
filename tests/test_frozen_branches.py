"""CPU contracts for one-forward branch tapes and preactivation interventions."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from experiments.frozen_branches import BranchTape, branch_intervention, record_branches

_MODES = ("sham", "relu", "pool", "both")


class _Reused(nn.Module):
    def __init__(self, *, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.alias = self.relu
        self.pool = nn.MaxPool2d(2, stride=1)
        self.p = nn.Parameter(torch.tensor([0.7, -0.2, 1.1, -0.4], dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        a = self.relu(self.p[0] * x + self.p[1])
        b = self.alias(self.p[2] * a + self.p[3])
        return self.pool(b)


def _snapshot(model: nn.Module) -> list[tuple]:
    return [
        (
            name,
            module,
            module.training,
            dict(module._forward_pre_hooks),
            dict(module._forward_hooks),
            dict(module._backward_hooks),
            "forward" in module.__dict__,
            module.__dict__.get("forward"),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    ]


def test_tape_payload_weights_only_roundtrip_and_small_summary() -> None:
    model = _Reused()
    x = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    tape = record_branches(model, lambda: model(x))
    payload = tape.to_payload()
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = BranchTape.from_payload(torch.load(stream, weights_only=True))
    summary = restored.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["calls"] == 3
    assert summary["relu_calls"] == 2
    assert summary["pool_calls"] == 1
    assert summary["elements"] == 22
    assert summary["bytes"] == 50  # 18 bools and four int64 spatial indices.
    for event in payload["events"]:
        assert event["pattern"].device.type == "cpu"
        assert event["pattern"].dtype in (torch.bool, torch.int64)
    with branch_intervention(model, restored, "both"):
        torch.testing.assert_close(
            model(x), torch.tensor([[[[2.46, 3.23], [4.77, 5.54]]]], dtype=x.dtype)
        )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("mode", _MODES)
def test_float64_parameter_gradient_and_dense_hessian_match_native_reference(
    inplace: bool, mode: str
) -> None:
    model = _Reused(inplace=inplace)
    x = torch.tensor(
        [[[[-2.0, 0.1, 1.2], [2.3, 3.4, -0.7], [1.8, -1.1, 0.6]]]], dtype=torch.float64
    )
    base = model.p.detach().clone()
    zero = torch.zeros_like(base)

    def reference(delta: Tensor) -> Tensor:
        p = base + delta
        a = F.relu(p[0] * x + p[1])
        b = F.relu(p[2] * a + p[3])
        return F.max_pool2d(b, 2, stride=1).square().sum()

    def loss(delta: Tensor) -> Tensor:
        return functional_call(model, {"p": base + delta}, (x,)).square().sum()

    expected_loss = reference(zero)
    expected_grad = torch.autograd.functional.jacobian(reference, zero)
    expected_hessian = torch.autograd.functional.hessian(reference, zero)
    tape = record_branches(model, lambda: loss(zero))
    with branch_intervention(model, tape, mode):
        torch.testing.assert_close(loss(zero), expected_loss, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(loss, zero),
            expected_grad,
            rtol=1e-12,
            atol=1e-12,
        )
        actual_hessian = torch.autograd.functional.hessian(loss, zero)
        torch.testing.assert_close(
            actual_hessian, expected_hessian, rtol=1e-12, atol=1e-12
        )
        for direction in (
            torch.ones_like(zero),
            torch.tensor([0.3, -0.9, 0.2, 0.8], dtype=zero.dtype),
        ):
            _, hvp = torch.autograd.functional.hvp(loss, zero, direction)
            torch.testing.assert_close(
                hvp, expected_hessian @ direction, rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize(
    "mode, expected", [("sham", 9.0), ("relu", 0.0), ("pool", 0.0), ("both", -5.0)]
)
def test_displaced_input_is_masked_or_gathered_before_native_operation(
    mode: str, expected: float
) -> None:
    model = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2))
    reference = torch.tensor([[[[1.0, -2.0], [-3.0, -4.0]]]])
    tape = record_branches(model, lambda: model(reference.clone()))
    displaced = torch.tensor([[[[-5.0, 9.0], [8.0, 7.0]]]], requires_grad=True)
    original = displaced.detach().clone()
    with branch_intervention(model, tape, mode):
        actual = model(displaced)
    assert actual.item() == expected
    torch.testing.assert_close(displaced, original)


def test_reused_inplace_relu_equal_shapes_selects_each_calls_own_mask() -> None:
    class IndependentCalls(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: Tensor) -> Tensor:
            return self.relu(x * 1) + 2 * self.relu(-x)

    model = IndependentCalls()
    tape = record_branches(model, lambda: model(torch.tensor([1.0, -1.0])))
    assert tape.events[0].pattern.tolist() == [True, False]
    assert tape.events[1].pattern.tolist() == [False, True]
    with branch_intervention(model, tape, "relu"):
        for _ in range(2):
            torch.testing.assert_close(
                model(torch.tensor([-3.0, 4.0])), torch.tensor([-3.0, -8.0])
            )


def test_fixed_operations_do_not_evaluate_native_relu_or_pool() -> None:
    model = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
    x = torch.tensor([[[[1.0, -2.0], [-3.0, -4.0]]]])
    tape = record_branches(model, lambda: model(x))
    with branch_intervention(model, tape, "both"):
        with (
            patch.object(F, "relu", side_effect=AssertionError("native ReLU ran")),
            patch.object(
                F, "max_pool2d", side_effect=AssertionError("native pool ran")
            ),
        ):
            assert model(-x).item() == -1.0


@pytest.mark.parametrize(
    "x, expected_indices",
    [
        ([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [0, 1, 3, 4]),
        ([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]], [4, 4, 4, 4]),
    ],
)
def test_pool_ties_and_repeated_winners_preserve_selected_ad_jet(
    x: list[list[float]], expected_indices: list[int]
) -> None:
    model = nn.MaxPool2d(2, stride=1, return_indices=True)
    input = torch.tensor(x, dtype=torch.float64).reshape(1, 1, 3, 3)
    tape = record_branches(model, lambda: model(input))
    assert tape.events[0].module_path == ""
    assert tape.events[0].pattern.reshape(-1).tolist() == expected_indices

    def loss(value: Tensor) -> Tensor:
        return model(value)[0].square().sum()

    expected_g = torch.autograd.functional.jacobian(loss, input)
    expected_h = torch.autograd.functional.hessian(loss, input)
    with branch_intervention(model, tape, "pool"):
        actual, indices = model(input)
        assert indices.reshape(-1).tolist() == expected_indices
        torch.testing.assert_close(
            actual,
            torch.tensor(x, dtype=input.dtype)
            .reshape(1, 1, 3, 3)
            .flatten(2)
            .gather(2, indices.flatten(2))
            .reshape(1, 1, 2, 2),
        )
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(loss, input), expected_g, rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(loss, input), expected_h, rtol=0, atol=0
        )
    diagonal = expected_h.reshape(9, 9).diagonal()
    for index in range(9):
        assert diagonal[index].item() == 2 * expected_indices.count(index)


def test_relu_zero_uses_native_selected_zero_derivative_and_out_of_place_sham() -> None:
    model = nn.ReLU(inplace=True)
    input = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    tape = record_branches(model, lambda: model(input.clone()))
    assert tape.events[0].pattern.tolist() == [False, False, True]
    for mode in ("sham", "relu", "both"):
        value = input.clone().requires_grad_()
        with branch_intervention(model, tape, mode):
            output = model(value)
            (gradient,) = torch.autograd.grad(output.sum(), value)
        torch.testing.assert_close(value.detach(), input)
        torch.testing.assert_close(
            gradient, torch.tensor([0.0, 0.0, 1.0], dtype=input.dtype)
        )


class _Ordered(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.ReLU()
        self.b = nn.ReLU()
        self.order: tuple[str, ...] = ("a", "b", "a")

    def forward(self, x: Tensor) -> Tensor:
        for name in self.order:
            x = getattr(self, name)(x)
        return x


@pytest.mark.parametrize("order", [("b", "a", "a"), ("a", "b"), ("a", "b", "a", "a")])
def test_event_order_missing_and_extra_calls_are_rejected_even_for_equal_shapes(
    order: tuple[str, ...],
) -> None:
    model = _Ordered()
    input = torch.ones(2, 3)
    tape = record_branches(model, lambda: model(input))
    model.order = order
    before = _snapshot(model)
    with pytest.raises(RuntimeError):
        with branch_intervention(model, tape, "both"):
            model(input)
    assert _snapshot(model) == before


def test_root_counters_reset_and_shapes_are_checked_on_every_forward() -> None:
    model = _Ordered()
    input = torch.ones(2, 3)
    tape = record_branches(model, lambda: model(input))
    with branch_intervention(model, tape, "relu"):
        for _ in range(3):
            torch.testing.assert_close(model(-input), torch.zeros_like(input) - input)
        with pytest.raises(RuntimeError):
            model(torch.ones(3, 2))
        # A caught failure does not poison the following root forward.
        torch.testing.assert_close(model(input), input)


@pytest.mark.parametrize("failure", ["record", "body", "forward"])
def test_cleanup_restores_module_objects_overrides_modes_and_hooks_after_errors(
    failure: str,
) -> None:
    model = _Reused()
    model.train()
    model.pool.eval()
    calls = []
    handles = [
        model.register_forward_pre_hook(lambda *_: calls.append("root_pre")),
        model.relu.register_forward_hook(lambda *_: calls.append("relu_post")),
        model.pool.register_full_backward_hook(lambda *_: None),
    ]
    # Preserve an existing instance-level forward override, not only class methods.
    model.pool.forward = model.pool.forward  # type: ignore[method-assign]
    input = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    before = _snapshot(model)

    def bad_forward() -> Tensor:
        model(input)
        raise ValueError("sentinel failure")

    if failure == "record":
        with pytest.raises(ValueError):
            record_branches(model, bad_forward)
    else:
        tape = record_branches(model, lambda: model(input))
        with pytest.raises(ValueError):
            with branch_intervention(model, tape, "both"):
                if failure == "forward":
                    bad_forward()
                raise ValueError("sentinel failure")
    assert _snapshot(model) == before
    assert "root_pre" in calls and "relu_post" in calls
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("root_calls", [0, 2])
def test_reference_rejects_other_than_one_root_forward(root_calls: int) -> None:
    model = nn.Sequential(nn.ReLU())
    before = _snapshot(model)

    def forward() -> None:
        for _ in range(root_calls):
            model(torch.ones(2))

    with pytest.raises(RuntimeError):
        record_branches(model, forward)
    assert _snapshot(model) == before


@pytest.mark.parametrize(
    "pool", [nn.MaxPool1d(2), nn.MaxPool3d(2), nn.AdaptiveMaxPool2d(1)]
)
def test_unsupported_pool_modules_fail_explicitly_without_patching(
    pool: nn.Module,
) -> None:
    model = nn.Sequential(nn.ReLU(), pool)
    before = _snapshot(model)
    with pytest.raises((ValueError, NotImplementedError)):
        record_branches(model, lambda: model(torch.ones(1, 1, 4, 4)))
    assert _snapshot(model) == before


def test_unbatched_pool_is_explicitly_rejected() -> None:
    model = nn.MaxPool2d(2)
    with pytest.raises((ValueError, NotImplementedError)):
        record_branches(model, lambda: model(torch.ones(1, 4, 4)))


@pytest.mark.parametrize("mode", _MODES)
def test_train_batchnorm_buffers_remain_callers_responsibility(mode: str) -> None:
    model = (
        nn.Sequential(
            nn.Conv2d(1, 2, 1, bias=True),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        .double()
        .train()
    )
    x = torch.arange(18, dtype=torch.float64).reshape(2, 1, 3, 3) / 7
    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    original = {name: value.detach().clone() for name, value in model.named_buffers()}
    before = _snapshot(model)

    def forward() -> Tensor:
        buffers = {name: value.clone() for name, value in original.items()}
        return functional_call(model, {**params, **buffers}, (x,))

    def loss(delta: Tensor) -> Tensor:
        state = {name: value.clone() for name, value in original.items()}
        offset = 0
        for name, value in params.items():
            state[name] = value + delta[offset : offset + value.numel()].view_as(value)
            offset += value.numel()
        return functional_call(model, state, (x,)).square().sum()

    zero = torch.zeros(sum(value.numel() for value in params.values()), dtype=x.dtype)
    expected = forward()
    expected_g = torch.autograd.functional.jacobian(loss, zero)
    expected_h = torch.autograd.functional.hessian(loss, zero)
    tape = record_branches(model, forward)
    with branch_intervention(model, tape, mode):
        for _ in range(2):
            torch.testing.assert_close(forward(), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(loss, zero),
            expected_g,
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(loss, zero),
            expected_h,
            rtol=1e-11,
            atol=1e-11,
        )
        assert model[1].training
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert _snapshot(model) == before


def test_invalid_mode_and_nested_intervention_do_not_leak_patches() -> None:
    model = nn.Sequential(nn.ReLU())
    input = torch.ones(2)
    tape = record_branches(model, lambda: model(input))
    before = _snapshot(model)
    with pytest.raises(ValueError):
        with branch_intervention(model, tape, "native"):
            pass
    assert _snapshot(model) == before
    with branch_intervention(model, tape, "both"):
        with pytest.raises(RuntimeError):
            with branch_intervention(model, tape, "relu"):
                pass
        torch.testing.assert_close(model(-input), -input)
    assert _snapshot(model) == before


def test_payload_rejects_float_masks_and_invalid_pool_indices() -> None:
    model = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
    tape = record_branches(model, lambda: model(torch.ones(1, 1, 2, 2)))
    bad = tape.to_payload()
    bad["events"][0]["pattern"] = torch.ones(1, 1, 2, 2)
    with pytest.raises((TypeError, ValueError)):
        BranchTape.from_payload(bad)
    bad = tape.to_payload()
    bad["events"][1]["pattern"].fill_(4)
    with pytest.raises((TypeError, ValueError)):
        BranchTape.from_payload(bad)


def test_pool_padding_only_windows_are_explicitly_rejected() -> None:
    model = nn.MaxPool2d(2, stride=1, padding=1, dilation=2)
    before = _snapshot(model)
    with pytest.raises((ValueError, NotImplementedError)):
        record_branches(model, lambda: model(torch.ones(1, 1, 1, 1)))
    assert _snapshot(model) == before


@pytest.mark.parametrize(
    "pool",
    [
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.MaxPool2d(
            (2, 3), stride=(1, 2), padding=(1, 1), dilation=(1, 2), ceil_mode=True
        ),
    ],
)
def test_pool_geometry_and_noncontiguous_nchw_keep_channels_separate(
    pool: nn.Module,
) -> None:
    input = torch.arange(120, dtype=torch.float64).reshape(2, 2, 5, 6).transpose(2, 3)
    assert not input.is_contiguous()
    expected = pool(input)
    expected_g = torch.autograd.functional.jacobian(
        lambda x: pool(x).square().sum(), input
    )
    tape = record_branches(pool, lambda: pool(input))
    with branch_intervention(pool, tape, "pool"):
        torch.testing.assert_close(pool(input), expected, rtol=0, atol=0)
        actual_g = torch.autograd.functional.jacobian(
            lambda x: pool(x).square().sum(), input
        )
    torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)


def test_reused_pool_with_equal_call_shapes_keeps_distinct_winners() -> None:
    class ReusedPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.MaxPool2d(3, stride=1, padding=1)

        def forward(self, x: Tensor) -> Tensor:
            return self.pool(x) + self.pool(-x)

    model = ReusedPool()
    input = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    tape = record_branches(model, lambda: model(input))
    assert [e.call_index for e in tape.events] == [0, 1]
    assert tape.events[0].input_shape == tape.events[1].input_shape
    assert not torch.equal(tape.events[0].pattern, tape.events[1].pattern)
    with branch_intervention(model, tape, "pool"):
        for _ in range(2):
            torch.testing.assert_close(
                model(-input),
                torch.tensor(
                    [[[[-4.0, -5.0, -4.0], [-7.0, -8.0, -7.0], [-4.0, -5.0, -4.0]]]],
                    dtype=input.dtype,
                ),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("stage", ["record", "intervention"])
def test_internal_forward_exception_restores_everything_and_can_be_caught(
    stage: str,
) -> None:
    class Raises(_Reused):
        fail = False

        def forward(self, x: Tensor) -> Tensor:
            result = super().forward(x)
            if self.fail:
                raise ValueError("inside root")
            return result

    model = Raises()
    input = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    before = _snapshot(model)
    if stage == "record":
        model.fail = True
        with pytest.raises(ValueError):
            record_branches(model, lambda: model(input))
    else:
        tape = record_branches(model, lambda: model(input))
        with branch_intervention(model, tape, "both"):
            model.fail = True
            with pytest.raises(ValueError):
                model(input)
            model.fail = False
            assert model(input).shape == (1, 1, 2, 2)
    assert _snapshot(model) == before


def test_sham_does_not_reproduce_inplace_alias_side_effects() -> None:
    class Aliased(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: Tensor) -> Tensor:
            aliased = x * 1
            return self.relu(aliased) + aliased

    model = Aliased()
    input = torch.tensor([-2.0, 3.0])
    tape = record_branches(model, lambda: model(input))
    torch.testing.assert_close(model(input), torch.tensor([0.0, 6.0]))
    with branch_intervention(model, tape, "sham"):
        # This is intentionally out-of-place. Caller must reject such a model
        # through its mandatory native/sham equivalence check, not hide the alias.
        torch.testing.assert_close(model(input), torch.tensor([-2.0, 6.0]))
