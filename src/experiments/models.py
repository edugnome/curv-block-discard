"""Neural-network models with reusable segment and parameter-group interfaces.

Segmented models expose ordered activation boundaries through get_segments().
Parameter groups partition trainable weights for the curvature measurements.
DAG models additionally expose named intermediate nodes and activations.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
from torchvision.models.resnet import BasicBlock, ResNet

_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class _MLPBlock(nn.Module):
    """Linear -> [LayerNorm] -> Activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        use_layernorm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        self.linear = _spectral_norm(linear) if use_spectral_norm else linear
        self.ln = nn.LayerNorm(out_features) if use_layernorm else nn.Identity()
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.ln(self.linear(x)))


class PlainMLP(nn.Module):
    """Plain (vanilla) MLP without skip-connections.

    Architecture:
      flatten -> Linear(input_dim, width) + Act -> [Linear(width,width)+Act] x depth -> head

    The first layer (projection) maps the input to the target width.
    Measurement points are the outputs of each block (all dim = width).
    """

    def __init__(
        self,
        input_dim: int = 3072,
        width: int = 64,
        depth: int = 8,
        num_classes: int = 10,
        activation: str = "relu",
        use_layernorm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        act_fn = _ACTIVATION_MAP[activation]

        self.projection = _MLPBlock(
            input_dim,
            width,
            act_fn(),
            use_layernorm=use_layernorm,
            use_spectral_norm=use_spectral_norm,
        )
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(
                _MLPBlock(
                    width,
                    width,
                    act_fn(),
                    use_layernorm=use_layernorm,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(width, num_classes)
        self._depth = depth

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_layer_names(self) -> list[str]:
        return [f"block_{i}" for i in range(self._depth)]

    def get_segments(self) -> list[Callable[[Tensor], Tensor]]:
        """segments[0]: x -> f_0, segments[i]: f_{i-1} -> f_i, segments[N]: f_{N-1} -> logits."""
        segs: list[Callable[[Tensor], Tensor]] = []

        # Segment 0: flatten + projection + block_0
        proj = self.projection
        first_block = self.blocks[0]

        def seg0(x: Tensor, p: nn.Module = proj, b: nn.Module = first_block) -> Tensor:
            return b(p(x.flatten(1)))

        segs.append(seg0)

        # Segments 1..depth-1
        for i in range(1, self._depth):
            block = self.blocks[i]
            segs.append(block)

        # Segment N: head
        segs.append(self.head)
        return segs


class _ResidualMLPBlock(nn.Module):
    """Linear -> Act + residual skip (identity, since all blocks are width->width)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = activation
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha * self.act(self.linear(x)) + x


class ResidualMLP(nn.Module):
    """MLP with skip-connections at each block.

    Architecture:
      flatten -> Linear(input_dim, width) + Act -> [Linear(width,width)+Act+skip] x depth -> head

    The first layer (projection) - shared with PlainMLP, maps input to width.
    All blocks are width->width with an identity skip each.
    Measurement points are the outputs of each block (all dim = width).

    Skip-connections preserve long-range resonance R(v,w)
    as dist(v,w) grows, according to Theorem S8 of the paper.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        width: int = 64,
        depth: int = 8,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_fn = _ACTIVATION_MAP[activation]

        self.projection = _MLPBlock(input_dim, width, act_fn())
        layers: list[nn.Module] = []
        alpha = 1.0 / depth**0.5
        for _ in range(depth):
            layers.append(_ResidualMLPBlock(width, width, act_fn(), alpha=alpha))
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(width, num_classes)
        self._depth = depth

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_layer_names(self) -> list[str]:
        return [f"block_{i}" for i in range(self._depth)]

    def get_segments(self) -> list[Callable[[Tensor], Tensor]]:
        segs: list[Callable[[Tensor], Tensor]] = []

        proj = self.projection
        first_block = self.blocks[0]

        def seg0(x: Tensor, p: nn.Module = proj, b: nn.Module = first_block) -> Tensor:
            return b(p(x.flatten(1)))

        segs.append(seg0)

        for i in range(1, self._depth):
            block = self.blocks[i]
            segs.append(block)

        segs.append(self.head)
        return segs


class BottleneckMLP(nn.Module):
    """Plain MLP with a single narrow (bottleneck) layer.

    Architecture:
      flatten -> projection(input_dim, base_width) + Act
        -> [Linear(w_i, w_{i+1}) + Act] x depth -> head

    All layers have width base_width, except one at
    bottleneck_position, where the width = bottleneck_width.
    Measurement points are the outputs of each block.

    If bottleneck_width == base_width, the model is identical to PlainMLP.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        base_width: int = 256,
        bottleneck_width: int = 64,
        bottleneck_position: int = -1,
        depth: int = 8,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_fn = _ACTIVATION_MAP[activation]

        if bottleneck_position < 0:
            bottleneck_position = depth // 2
        self._bottleneck_position = bottleneck_position

        # out_dims[i] - output width of block i
        out_dims = [base_width] * depth
        out_dims[bottleneck_position] = bottleneck_width

        # in_dims[i] - input width of block i
        in_dims = [base_width] + out_dims[:-1]

        self.projection = _MLPBlock(input_dim, base_width, act_fn())
        layers: list[nn.Module] = []
        for i in range(depth):
            layers.append(_MLPBlock(in_dims[i], out_dims[i], act_fn()))
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(out_dims[-1], num_classes)
        self._depth = depth

    @property
    def bottleneck_position(self) -> int:
        return self._bottleneck_position

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_layer_names(self) -> list[str]:
        return [f"block_{i}" for i in range(self._depth)]

    def get_segments(self) -> list[Callable[[Tensor], Tensor]]:
        segs: list[Callable[[Tensor], Tensor]] = []

        proj = self.projection
        first_block = self.blocks[0]

        def seg0(x: Tensor, p: nn.Module = proj, b: nn.Module = first_block) -> Tensor:
            return b(p(x.flatten(1)))

        segs.append(seg0)

        for i in range(1, self._depth):
            block = self.blocks[i]
            segs.append(block)

        segs.append(self.head)
        return segs


class DiamondMLP(nn.Module):
    """Two-branch (diamond) MLP to activate all terms of formula (1).

    Architecture:
      flatten -> stem: Linear(input_dim, width)+Act
        |-- Branch A: [Linear(width, width)+Act] x branch_depth
        |-- Branch B: [Linear(width, width)+Act] x branch_depth
                    -> merge
      -> head: Linear(width, num_classes) -> logits

    Merge modes:
      - ``sum``:  (h_A + h_B) / sqrt(2)  - linear, T_{merge;A,B} = 0.
      - ``cat``:  Act(Linear(2*width, width)(concat(h_A, h_B)))  - nonlinear,
        T_{merge;A,B} != 0 when sigma'' != 0.

    The ``forward_with_intermediates`` method returns named intermediate
    activations for computing Hessian blocks via autograd without
    assuming a sequential topology.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        width: int = 32,
        branch_depth: int = 2,
        num_classes: int = 10,
        activation: str = "relu",
        merge: str = "sum",
    ) -> None:
        super().__init__()
        act_fn = _ACTIVATION_MAP[activation]

        self.stem = _MLPBlock(input_dim, width, act_fn())

        self.branch_a = nn.ModuleList(
            [_MLPBlock(width, width, act_fn()) for _ in range(branch_depth)]
        )
        self.branch_b = nn.ModuleList(
            [_MLPBlock(width, width, act_fn()) for _ in range(branch_depth)]
        )

        self._merge_type = merge
        if merge == "cat":
            self.merge_layer = nn.Sequential(
                nn.Linear(2 * width, width),
                act_fn(),
            )
        else:
            self._inv_sqrt2 = 1.0 / math.sqrt(2.0)

        self.head = nn.Linear(width, num_classes)
        self._branch_depth = branch_depth

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x.flatten(1))
        h_a, h_b = h, h
        for block in self.branch_a:
            h_a = block(h_a)
        for block in self.branch_b:
            h_b = block(h_b)
        h = self._merge(h_a, h_b)
        return self.head(h)

    def forward_with_intermediates(
        self,
        x: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass with intermediate activations saved.

        Returns:
            logits: (B, num_classes)
            intermediates: {node_name: Tensor} for each DAG node.
        """
        intermediates: dict[str, Tensor] = {}

        h = self.stem(x.flatten(1))
        h_a, h_b = h, h
        for i, block in enumerate(self.branch_a):
            h_a = block(h_a)
            intermediates[f"A_{i}"] = h_a
        for i, block in enumerate(self.branch_b):
            h_b = block(h_b)
            intermediates[f"B_{i}"] = h_b

        h = self._merge(h_a, h_b)
        intermediates["merge"] = h
        logits = self.head(h)
        return logits, intermediates

    def get_node_names(self) -> list[str]:
        """Names of intermediate DAG nodes (for Hessian pairs)."""
        names: list[str] = []
        for i in range(self._branch_depth):
            names.append(f"A_{i}")
        for i in range(self._branch_depth):
            names.append(f"B_{i}")
        names.append("merge")
        return names

    def _merge(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        if self._merge_type == "cat":
            return self.merge_layer(torch.cat([h_a, h_b], dim=-1))
        return (h_a + h_b) * self._inv_sqrt2


# ======================================================================
# ResNet-18 (conv) - Experiment 6
# ======================================================================


def _disable_shortcuts(base: nn.Module) -> None:
    """Replaces forward of all BasicBlocks: removes the identity shortcut.

    Original: out = relu(conv2(bn2(conv1(bn1(x)))) + identity).
    Here:     out = relu(conv2(bn2(conv1(bn1(x))))).

    Spatial downsampling is handled via stride=2 in conv1 (layer2-4[0]);
    no shortcut is needed for dimension matching.
    """

    def _make_plain_forward(block: BasicBlock):
        def forward(x: Tensor) -> Tensor:
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)
            out = block.conv2(out)
            out = block.bn2(out)
            out = block.relu(out)
            return out

        return forward

    for module in base.modules():
        if isinstance(module, BasicBlock):
            module.forward = _make_plain_forward(module)  # type: ignore[assignment]
            module.downsample = None


class _WideResNet(ResNet):
    """torchvision BasicBlock ResNet with every stage width scaled by ``width_mult``.

    torchvision forbids width scaling for ``BasicBlock`` (``base_width`` must stay 64) and
    hardcodes the stage widths 64/128/256/512, so the stock factory cannot build a 0.5x or
    2x ResNet-18. This rebuilds the stem and the four stages at the scaled widths while
    reusing the official ``BasicBlock`` and the inherited ``_make_layer`` / ``forward`` (the
    residual logic is unchanged); at ``width_mult = 1.0`` it reproduces the standard
    ResNet-18 channel counts exactly.
    """

    def __init__(self, layers: list[int], num_classes: int, width_mult: float) -> None:
        nn.Module.__init__(self)
        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        widths = [
            round(64 * width_mult),
            round(128 * width_mult),
            round(256 * width_mult),
            round(512 * width_mult),
        ]
        self.inplanes = widths[0]
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, widths[0], layers[0])
        self.layer2 = self._make_layer(BasicBlock, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3] * BasicBlock.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SegmentedResNet18(nn.Module):
    """ResNet-18 with segmentation for inter-layer Hessian computation.

    Wrapper around ``torchvision.models.resnet18(num_classes)``.
    Segments (6 total):
      seg0: conv1 -> bn1 -> relu -> maxpool  (3,32,32) -> (64,8,8)
      seg1: layer1  (two BasicBlocks)        (64,8,8)  -> (64,8,8)
      seg2: layer2                           (64,8,8)  -> (128,4,4)
      seg3: layer3                           (128,4,4) -> (256,2,2)
      seg4: layer4                           (256,2,2) -> (512,1,1)
      seg5: avgpool -> flatten -> fc         (512,1,1) -> (10,)

    Measurement points (5): stem, layer1, layer2, layer3, layer4.
    """

    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self._base = self._make_base(num_classes)
        self._activation = activation
        if activation != "relu":
            self._replace_activations(activation)

    def _make_base(self, num_classes: int) -> ResNet:
        """Build the wrapped torchvision backbone (overridden for other depths)."""
        return tv_models.resnet18(num_classes=num_classes)

    def _replace_activations(self, activation: str) -> None:
        """Replaces all ReLU activations in the base model with the specified activation."""
        act_cls = _ACTIVATION_MAP[activation]
        # Top-level
        self._base.relu = act_cls()
        # BasicBlocks in all layer{1..4}
        for layer in (
            self._base.layer1,
            self._base.layer2,
            self._base.layer3,
            self._base.layer4,
        ):
            for block in layer:
                block.relu = act_cls()

    def forward(self, x: Tensor) -> Tensor:
        return self._base(x)

    def get_layer_names(self) -> list[str]:
        return ["stem", "layer1", "layer2", "layer3", "layer4"]

    def get_segments(self) -> list[Callable[[Tensor], Tensor]]:
        b = self._base
        segs: list[Callable[[Tensor], Tensor]] = []

        def seg0(x: Tensor) -> Tensor:
            h = b.conv1(x)
            h = b.bn1(h)
            h = b.relu(h)
            h = b.maxpool(h)
            return h

        segs.append(seg0)
        segs.append(b.layer1)
        segs.append(b.layer2)
        segs.append(b.layer3)
        segs.append(b.layer4)

        def seg5(x: Tensor) -> Tensor:
            h = b.avgpool(x)
            h = torch.flatten(h, 1)
            h = b.fc(h)
            return h

        segs.append(seg5)
        return segs

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Trainable parameters grouped by measurement segment.

        One group per `get_segments` measurement point - stem (conv1 + bn1), each of
        layer1-4, and the classifier head - in segment (group-major) order. This is the
        partition the parameter-space estimator (`ParamBlockEstimator`) and a K-FAC
        provider consume, so a block-diagonal preconditioner's blocks line up with the
        activation measurement points. BatchNorm parameters travel with their segment;
        a K-FAC backend that does not factor them falls back to an identity block.
        """
        b = self._base
        return {
            "stem": list(b.conv1.parameters()) + list(b.bn1.parameters()),
            "layer1": list(b.layer1.parameters()),
            "layer2": list(b.layer2.parameters()),
            "layer3": list(b.layer3.parameters()),
            "layer4": list(b.layer4.parameters()),
            "head": list(b.fc.parameters()),
        }


class SegmentedPlainResNet18(SegmentedResNet18):
    """ResNet-18 without skip-connections (plain conv-bn-act chain).

    Disables the identity shortcut in all BasicBlocks,
    preserving the parameterization (conv + bn) and spatial downsampling
    (stride=2 in conv1 of layer2-4[0]).
    """

    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__(num_classes=num_classes, activation=activation)
        _disable_shortcuts(self._base)


class SegmentedResNet50(SegmentedResNet18):
    """ResNet-50 sharing SegmentedResNet18's torchvision-ResNet segmentation contract.

    Same six segments / five measurement points (stem, layer1-4, head); only the
    backbone depth (Bottleneck blocks, 4x channel expansion) and optional ImageNet
    pretraining differ. On the standard 224x224 input the segment outputs are
      stem (64,56,56) -> layer1 (256,56,56) -> layer2 (512,28,28)
      -> layer3 (1024,14,14) -> layer4 (2048,7,7) -> head (num_classes,).

    Primary use is fine-grained full finetuning: ``pretrained=True`` loads
    IMAGENET1K_V2 weights and swaps the 1000-way classifier for a fresh ``num_classes``
    head.
    """

    def __init__(
        self,
        num_classes: int = 196,
        *,
        pretrained: bool = True,
        activation: str = "relu",
    ) -> None:
        self._pretrained = pretrained
        super().__init__(num_classes=num_classes, activation=activation)

    def _make_base(self, num_classes: int) -> ResNet:
        if self._pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            checkpoint = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / Path(urlparse(weights.url).path).name
            )
            if (
                not checkpoint.is_file()
                and os.environ.get("DAG_HESSE_ALLOW_DOWNLOADS") != "1"
            ):
                raise FileNotFoundError(
                    f"Pretrained ResNet-50 weights are not cached at {checkpoint}. "
                    "Populate the Torch cache or explicitly enable --allow-downloads "
                    "(DAG_HESSE_ALLOW_DOWNLOADS=1)."
                )
            base = tv_models.resnet50(weights=weights)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            return base
        return tv_models.resnet50(num_classes=num_classes)


class SegmentedResNet34(SegmentedResNet18):
    """ResNet-34 sharing SegmentedResNet18's segmentation (deeper BasicBlock stack).

    Same six segments / five measurement points (stem, layer1-4, head); only the depth
    differs (BasicBlock layers [3, 4, 6, 3] vs [2, 2, 2, 2]). Provides the depth rung of
    the width/depth sweep (`Exp8Config.a1_width`).
    """

    def _make_base(self, num_classes: int) -> ResNet:
        return tv_models.resnet34(num_classes=num_classes)


class SegmentedWideResNet18(SegmentedResNet18):
    """ResNet-18 with all stage widths scaled by ``width_mult`` (the width axis).

    Shares SegmentedResNet18's segmentation contract: the channel counts change but the
    six conv1 / bn1 / layer1-4 / fc measurement points do not, so `get_param_groups` and
    `get_segments` apply unchanged. ``width_mult = 1.0`` reproduces the standard ResNet-18.
    """

    def __init__(
        self,
        num_classes: int = 10,
        *,
        width_mult: float = 1.0,
        activation: str = "relu",
    ) -> None:
        self._width_mult = width_mult
        super().__init__(num_classes=num_classes, activation=activation)

    def _make_base(self, num_classes: int) -> ResNet:
        return _WideResNet([2, 2, 2, 2], num_classes, self._width_mult)


class SegmentedMLP(nn.Module):
    """Fully-connected classifier whose Linear layers are full-rank K-FAC/EKFAC blocks.

    A low input dimension keeps every Kronecker factor full rank at a modest batch, so
    EKFAC's eigenvalue correction is well conditioned and genuinely distinct from K-FAC -
    the clean, full-rank setting for the optimizer-agnostic overlay comparison
    (`Exp7Config.b1_mlp`), complementing the rank-deficient convolutional headline. Each
    Linear layer is one parameter group (group-major ``fc0 .. fc{depth-1}`` then ``head``),
    matching the layout the parameter-space estimator and the K-FAC / EKFAC providers share.
    """

    def __init__(
        self,
        *,
        in_dim: int = 64,
        hidden: int = 64,
        depth: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        widths = [in_dim] + [hidden] * depth
        self.blocks = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(depth)
        )
        self.head = nn.Linear(hidden, num_classes)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = x.flatten(1)
        for block in self.blocks:
            h = self.act(block(h))
        return self.head(h)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {
            f"fc{i}": list(block.parameters()) for i, block in enumerate(self.blocks)
        }
        groups["head"] = list(self.head.parameters())
        return groups


# ======================================================================
# Toy-Attention & ReLU-MLP - Experiment 5
# ======================================================================


class ToyAttentionModel(nn.Module):
    """Single-head self-attention for verification of H^T_{Q,K} != 0.

    Architecture:
      x (B, S, d) -> W_Q, W_K, W_V projections -> Softmax(QK^T/sqrt(d)) V
        -> mean-pool -> Linear -> y (B, 1)

    Softmax-attention has sigma''!=0, so the tensor term H^T_{Q,K}
    does not vanish (unlike ReLU-MLP).

    Named intermediate nodes for the Hessian:
      "Q": Q = x @ W_Q^T  (B, S, d)
      "K": K = x @ W_K^T  (B, S, d)
      "V": V = x @ W_V^T  (B, S, d)
      "attn_out": Softmax(QK^T/sqrt(d)) @ V  (B, S, d)
    """

    def __init__(
        self,
        d_model: int = 16,
        seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.scale = 1.0 / math.sqrt(d_model)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Regression head: mean-pool -> linear -> scalar
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, d) -> y: (B, 1)."""
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        pooled = attn_out.mean(dim=1)  # (B, d)
        return self.head(pooled)  # (B, 1)

    def forward_with_intermediates(
        self,
        x: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass with intermediate activations saved.

        Returns:
            output: (B, 1) - regression output.
            intermediates: {node_name: Tensor} for each node.
        """
        intermediates: dict[str, Tensor] = {}

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        intermediates["Q"] = Q
        intermediates["K"] = K
        intermediates["V"] = V

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        intermediates["attn_out"] = attn_out

        pooled = attn_out.mean(dim=1)
        output = self.head(pooled)
        return output, intermediates

    def get_node_names(self) -> list[str]:
        """Names of intermediate nodes for Hessian pair computation."""
        return ["Q", "K", "V", "attn_out"]


class ToyReluMLP(nn.Module):
    """Per-position ReLU-MLP control for verification of H^T = 0.

    Architecture mirrors per-token attention processing:
      x (B, S, d) -> 3x [Linear(d, d) + ReLU] (per-position, shared)
        -> mean-pool -> Linear -> y (B, 1)

    Parameters: 3(d^2 + d) + d + 1 = 833 at d=16  (vs 785 for attention).
    All nonlinearities are ReLU (sigma'' = 0 a.e.), so by Proposition 4:
    H^T_{block_0, block_1} = 0 and GN-Gap ~= 0.
    """

    def __init__(
        self,
        d_model: int = 16,
        seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.layer3 = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, d) -> y: (B, 1)."""
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(h))
        return self.head(h.mean(dim=1))

    def forward_with_intermediates(
        self,
        x: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward with intermediate activations saved (block_0, block_1)."""
        intermediates: dict[str, Tensor] = {}
        f0 = F.relu(self.layer1(x))  # (B, S, d)
        f1 = F.relu(self.layer2(f0))  # (B, S, d)
        f2 = F.relu(self.layer3(f1))  # (B, S, d)
        intermediates["block_0"] = f0
        intermediates["block_1"] = f1
        output = self.head(f2.mean(dim=1))  # (B, 1)
        return output, intermediates
