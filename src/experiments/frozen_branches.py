"""Temporary module-level ReLU/MaxPool2d interventions on a diagnostic model.

The caller owns functional parameters, fresh BN buffer copies, modes and AD-jet
validation. Only registered, standard nn.ReLU and NCHW nn.MaxPool2d operations
are observed, not functional calls or arbitrary graph nodes. Sham deliberately
uses out-of-place ReLU: alias-sensitive networks must pass the caller's
native/sham check. Use an otherwise idle model; contexts are not thread-safe,
reentrant, or compatible with compiled/checkpoint-recomputed forwards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class BranchEvent:
    """One call; indices address the flattened spatial input, separately per N,C."""

    module_path: str
    call_index: int
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    pattern: Tensor


@dataclass(frozen=True)
class BranchTape:
    """Detached bool/int64 patterns, in execution order (never floating activations)."""

    events: tuple[BranchEvent, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a CPU-only, weights_only=True-compatible independent payload."""
        return {
            "version": 1,
            "events": [
                {
                    "module_path": e.module_path,
                    "call_index": e.call_index,
                    "kind": e.kind,
                    "input_shape": e.input_shape,
                    "output_shape": e.output_shape,
                    "pattern": e.pattern.detach().cpu().clone(),
                }
                for e in self.events
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BranchTape:
        """Validate and copy the plain tensor payload; no custom pickle objects."""
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported BranchTape payload version")
        raw = payload.get("events")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("BranchTape events must be a sequence")
        events = []
        counts: dict[str, int] = {}
        kinds: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each branch event must be a dict")
            path, index, kind = (
                item.get("module_path"),
                item.get("call_index"),
                item.get("kind"),
            )
            if (
                not isinstance(path, str)
                or type(index) is not int
                or index != counts.get(path, 0)
                or kind not in ("relu", "pool")
                or kinds.get(path, kind) != kind
            ):
                raise ValueError("Invalid branch event identity or call sequence")
            shapes = []
            for name in ("input_shape", "output_shape"):
                shape = item.get(name)
                if not isinstance(shape, (list, tuple)) or any(
                    type(n) is not int or n < 0 for n in shape
                ):
                    raise ValueError("Invalid branch event shape")
                shapes.append(tuple(shape))
            input_shape, output_shape = shapes
            pattern = item.get("pattern")
            dtype = torch.bool if kind == "relu" else torch.int64
            if (
                not isinstance(pattern, Tensor)
                or pattern.device.type != "cpu"
                or pattern.layout != torch.strided
                or pattern.dtype != dtype
                or pattern.requires_grad
                or tuple(pattern.shape) != output_shape
            ):
                raise ValueError(f"Invalid {kind} pattern: expected CPU {dtype}")
            if kind == "relu" and input_shape != output_shape:
                raise ValueError("ReLU input/output shapes must match")
            if kind == "pool":
                if (
                    len(input_shape) != 4
                    or len(output_shape) != 4
                    or input_shape[:2] != output_shape[:2]
                    or min(input_shape[1:] + output_shape[1:], default=0) <= 0
                ):
                    raise ValueError("Pool patterns require NCHW shapes")
                spatial = input_shape[2] * input_shape[3]
                if bool(((pattern < 0) | (pattern >= spatial)).any()):
                    raise ValueError("Pool pattern indices out of range")
            events.append(
                BranchEvent(
                    path, index, kind, input_shape, output_shape, pattern.clone()
                )
            )
            counts[path] = index + 1
            kinds[path] = kind
        return cls(tuple(events))

    def summary(self) -> dict[str, int]:
        """Small JSON-safe storage summary, including calls of reused modules."""
        return {
            "calls": len(self.events),
            "relu_calls": sum(e.kind == "relu" for e in self.events),
            "pool_calls": sum(e.kind == "pool" for e in self.events),
            "elements": sum(e.pattern.numel() for e in self.events),
            "bytes": sum(
                e.pattern.numel() * e.pattern.element_size() for e in self.events
            ),
        }


_UNSUPPORTED_POOLS = (
    nn.MaxPool1d,
    nn.MaxPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
)
_MISSING = object()


class _Session:
    def __init__(
        self, model: nn.Module, tape: BranchTape | None, mode: str = "sham"
    ) -> None:
        self.model, self.mode = model, mode
        self.recording = tape is None
        self.events: list[BranchEvent] = [] if tape is None else list(tape.events)
        self.position = self.root_calls = 0
        self.active = False
        self.counts: dict[str, int] = {}
        self.patterns: dict[int, Tensor] = {}
        self.saved: list[tuple[nn.Module, Any]] = []
        self.modules: dict[str, nn.ReLU | nn.MaxPool2d] = {}
        for path, module in model.named_modules():
            if getattr(module.forward, "_branch_session", False):
                raise RuntimeError(
                    "A branch session is already active; nesting is unsupported"
                )
            if isinstance(module, _UNSUPPORTED_POOLS):
                raise NotImplementedError("Only NCHW nn.MaxPool2d is supported")
            if isinstance(module, (nn.ReLU, nn.MaxPool2d)):
                if type(module) not in (nn.ReLU, nn.MaxPool2d):
                    raise NotImplementedError(
                        "Custom ReLU/MaxPool2d subclasses are unsupported"
                    )
                self.modules[path] = module
        for event in self.events:
            target = self.modules.get(event.module_path)
            kind = "relu" if isinstance(target, nn.ReLU) else "pool"
            if target is None or kind != event.kind:
                raise RuntimeError(
                    "Branch event module path or operation does not match model"
                )

    def _patch(self, module: nn.Module, forward: Callable) -> None:
        self.saved.append((module, module.__dict__.get("forward", _MISSING)))
        forward._branch_session = True  # type: ignore[attr-defined]
        module.forward = forward  # type: ignore[method-assign]

    @contextmanager
    def installed(self) -> Iterator[None]:
        try:
            for path, module in self.modules.items():
                self._patch(module, self._operation(path, module))
            original = self.model.forward

            def root_forward(*args: Any, **kwargs: Any) -> Any:
                if self.active:
                    raise RuntimeError("Reentrant root forwards are unsupported")
                if self.recording and self.root_calls:
                    raise RuntimeError("Reference recording requires one root forward")
                self.root_calls += 1
                self.position = 0
                self.counts.clear()
                self.active = True
                try:
                    result = original(*args, **kwargs)
                    if not self.recording and self.position != len(self.events):
                        raise RuntimeError(
                            "Branch event sequence ended with missing calls"
                        )
                    return result
                finally:
                    self.active = False

            self._patch(self.model, root_forward)
            yield
        finally:
            for restored_module, previous in reversed(self.saved):
                if previous is _MISSING:
                    del restored_module.forward
                else:
                    restored_module.forward = previous
            self.saved.clear()

    def _operation(self, path: str, module: nn.ReLU | nn.MaxPool2d) -> Callable:
        kind = "relu" if isinstance(module, nn.ReLU) else "pool"

        def forward(input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            if not self.active:
                raise RuntimeError(
                    "Branch calls must occur inside the model root forward"
                )
            if not isinstance(input, Tensor):
                raise TypeError("Branch operations require tensor input")
            if kind == "pool" and input.ndim != 4:
                raise NotImplementedError(
                    "MaxPool2d intervention requires NCHW (4D) input"
                )
            call_index = self.counts.get(path, 0)
            self.counts[path] = call_index + 1
            shape = tuple(input.shape)
            event = None
            if not self.recording:
                if self.position >= len(self.events):
                    raise RuntimeError("Branch event sequence contains extra calls")
                event = self.events[self.position]
                if (path, call_index, kind) != (
                    event.module_path,
                    event.call_index,
                    event.kind,
                ):
                    raise RuntimeError(
                        "Branch event sequence or call identity does not match"
                    )
                if shape != event.input_shape:
                    raise RuntimeError(
                        "Branch event input shape does not match reference"
                    )
            frozen = not self.recording and self.mode in (kind, "both")
            if frozen:
                assert event is not None
                pattern = self.patterns.get(self.position, event.pattern)
                if pattern.device != input.device:
                    pattern = pattern.to(device=input.device)
                self.patterns[self.position] = pattern
                if kind == "relu":
                    output = input * pattern
                else:
                    indices = pattern
                    output = input.flatten(2).gather(2, indices.flatten(2))
                    output = output.reshape(event.output_shape)
            elif isinstance(module, nn.ReLU):
                if self.recording:
                    # Capture before native inplace ReLU mutates the input.
                    pattern = input.detach() > 0
                output = F.relu(input, inplace=self.recording and module.inplace)
            else:
                output, indices = F.max_pool2d(
                    input,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.ceil_mode,
                    return_indices=True,
                )
                if self.recording:
                    spatial = input.shape[2] * input.shape[3]
                    if bool(((indices < 0) | (indices >= spatial)).any()):
                        raise NotImplementedError(
                            "Pool windows without real input winners are unsupported"
                        )
                    pattern = indices.detach().clone()
            output_shape = tuple(output.shape)
            if self.recording:
                self.events.append(
                    BranchEvent(path, call_index, kind, shape, output_shape, pattern)
                )
            else:
                assert event is not None
                if output_shape != event.output_shape:
                    raise RuntimeError(
                        "Branch event output shape does not match reference"
                    )
            self.position += 1
            if isinstance(module, nn.MaxPool2d) and module.return_indices:
                # Do not expose the tape's storage as a mutable forward output.
                return output, indices.clone() if frozen else indices
            return output

        return forward


def record_branches(model: nn.Module, forward: Callable[[], Any]) -> BranchTape:
    """Call the supplied zero-displacement forward once and detach its branches.

    This does not switch mode, copy buffers, suppress gradients, or restore RNG.
    Temporary forward observers are removed even when the caller's forward raises.
    """
    session = _Session(model, None)
    with session.installed():
        forward()
        if session.root_calls != 1:
            raise RuntimeError("Reference recording requires one root forward")
    return BranchTape(tuple(session.events))


@contextmanager
def branch_intervention(
    model: nn.Module, tape: BranchTape, mode: str
) -> Iterator[None]:
    """Replay one reference tape for every root forward in sham/relu/pool/both mode.

    Module objects and existing hooks stay in place; only their forward operations
    are temporarily replaced. Fixed branches operate on the *current* input, not
    on an already rectified/pooled output. CPU payload patterns move lazily to the
    input device and are cached only for this context. Native runs outside it.
    """
    if mode not in ("sham", "relu", "pool", "both"):
        raise ValueError(
            "mode must be sham, relu, pool, or both (native is outside context)"
        )
    session = _Session(model, tape, mode)
    with session.installed():
        yield
