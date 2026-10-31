"""Experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class TrainingConfig:
    lr: float = 0.01
    batch_size: int = 128
    epochs: int = 50
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "none" | "cosine"
    # Pin every kernel to its deterministic variant for the whole run (bit-reproducible
    # arithmetic, at a throughput cost). The production profiles switch it off and keep
    # determinism only inside their measurements (`experiments.utils.deterministic_mode`).
    deterministic: bool = True
    # Stop training after this many optimizer steps in total (None = run all epochs);
    # a smoke-test budget, never set by the production profiles.
    max_steps: int | None = None


@dataclass
class HessianConfig:
    """Hessian computation mode selection."""

    mode: str = "exact"  # "exact" | "stochastic"
    n_probes: int = 30  # for stochastic mode
    n_power_iter: int = 20  # power iteration steps for ||H||_2
    hessian_batch_size: int = 32  # subsample from validation set


@dataclass
class Exp7Config:
    """Exp7: COUPLE-FAC overlay optimization configurations.

    COUPLE-FAC augments a block-diagonal K-FAC step with a low-rank cross-block
    saddle-free Newton correction restricted to the strongly-coupled subset
    S = {(v, w): C_{vw} > tau}, then accepts it only inside the relative trust region
    ||d_overlay|| <= tr_ratio ||d_KFAC|| and the damped exact-HVP quadratic model is
    no worse than the base model value up to the safeguard tolerance. An optional
    minibatch-loss check can also reject it.
    The method is amortized over two refresh periods: ``measure_period`` is the overlay
    refresh period T - the coupling C, the selection S, the pair subspaces U_vw and their
    reduced Hessians are rebuilt every T steps and reused in between (the payoff record
    is taken on the same steps) - and ``kfac_refresh_period`` is T_k, the period on which
    the K-FAC factors and their inverse are rebuilt; the gradient is fresh every step and
    the safeguard evaluates the damped model on every step.

    Two regimes share this schema (the "adaptive" fac/dag split):
      B1 (fine-grained finetune, the headline) - an ImageNet-pretrained ResNet-50 fully
        finetuned on Stanford Cars (196 classes, ~8k train) at 224x224; see `b1_cars`.
      B2 (large-batch from scratch, the control) - a ResNet trained from random init
        with large batches B in {1024, 4096} on ImageNet-32 (32x32); see `b2_largebatch`.

    The CIFAR loaders and `TrainingConfig` defaults are reused unchanged; only the
    `dataset`/resolution/optimizer knobs are adapted per regime.
    """

    # Data / backbone
    dataset: str = (
        "stanford_cars"  # "stanford_cars" | "cifar10" | "cifar100" | "imagenet32"
    )
    num_classes: int = 196
    model: str = "resnet50"  # backbone registry key
    pretrained: bool = True  # ImageNet init (B1); False for from-scratch (B2)
    image_size: int = 224
    augment: bool = True

    # COUPLE-FAC overlay + trust-region safeguard + K-FAC provider
    tau: float = 0.1  # coupling-selection threshold S = {C_{vw} > tau}
    overlay_rank: int = 4  # rank r <= 4 of the 2x2 block-Newton correction
    overlay_n_power_iter: int = 8  # subspace-iteration rounds for the overlay build
    overlay_max_pairs: int = 3  # |S| <= 3 strongest coupled pairs per refresh
    # T >= 20: the overlay refresh period - coupling C, selection S, subspaces U_vw and
    # reduced Hessians are rebuilt every T steps (and the payoff record is taken there).
    measure_period: int = 20
    # Hutchinson probes: the online estimate that ranks pairs for selection inside the
    # COUPLE-FAC hook (selection uses both threshold and rank) and Phi checkpoints.
    coupling_n_probes_online: int = 10
    coupling_n_probes: int = 30
    # Relative trust region of the no-harm safeguard: the corrected step is accepted only
    # if ||d_overlay|| <= tr_ratio * ||d_KFAC|| (a correction, not a replacement).
    tr_ratio: float = 2.0
    # Tikhonov damping lambda_0: the initial value of the ONE damping of a run, shared by
    # reference (`secondorder.Damping`): one value, read by the K-FAC / EKFAC inverse (at
    # its refresh), the overlay's reduced solve, the safeguard's model and the Newton-CG
    # oracle, tied by design so that one adaptation reaches all of them. It is not one
    # damped model: the K-FAC step damps its Kronecker factors, the EKFAC step its
    # eigenbasis exactly, the overlay's solve adds it to |lambda_i| on its reduced
    # Hessian, and only the safeguard's model and the oracle take the exact H + lambda I.
    # The default damping is shared across the provider, overlay, and model test.
    damping: float = 1.0
    # Levenberg-Marquardt adaptation of that damping (Martens & Grosse, 2015, Sec. 6.5),
    # applied at every payoff record from the base step's actual-to-predicted reduction
    # rho = (L(theta) - L(theta + d0)) / (-m(d0)): m(d0) >= 0 or rho < 1/4 multiplies
    # lambda by ``damping_factor``, rho > 3/4 divides it, then it is clamped to
    # ``damping_bounds``. Adaptation is optional and does not guarantee model
    # validity. ``sgd`` / ``adam`` are unaffected; ``newton_cg`` adapts on its own
    # step. The record fields ``damping`` and ``rho`` are written either way.
    damping_adapt: bool = False
    damping_factor: float = 2.0
    damping_bounds: tuple[float, float] = (1e-3, 1e3)
    # T_k: the K-FAC factors and their inverse are rebuilt only every this many steps
    # (the standard K-FAC amortization); the gradient they precondition is fresh.
    kfac_refresh_period: int = 20
    # Fixed validation minibatch on which Phi is measured at init / mid / final for
    # every method (the blind-spot diagnostic alongside the optimizer comparison).
    hessian_batch_size: int = 32
    # curvlinops Fisher estimate behind the K-FAC / EKFAC factors: the Monte-Carlo
    # Fisher with one sampled label per example (curvlinops' default). The exact
    # "type-2" Fisher costs one backward pass per class per factor pass - 196 on
    # Stanford Cars, and EKFAC runs two such passes per refresh (factors, then the
    # eigenvalue correction): 27 s + 86 s per refresh at batch 64, i.e. ~11 s/step at
    # T_k = 10 on top of the step itself. MC costs one backward per pass.
    fisher_type: str = "mc"
    # EKFAC rebuilds its eigenbasis only every this many steps - the eigendecomposition is
    # its dominant cost, and periodic factor refresh is standard practice (K-FAC's cheaper
    # factored step stays per-step). Only the EKFAC methods read this.
    ekfac_refresh_period: int = 10

    # Comparison + demonstration
    methods: list[str] = field(
        default_factory=lambda: [
            "sgd",
            "adam",
            "kfac",
            "ekfac",
            "kfac+overlay",
            "ekfac+overlay",
            "newton_cg",
        ]
    )
    target_acc: float = 0.9  # validation accuracy for the "speed to target" metric
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=0.01,
            batch_size=64,
            epochs=30,
            optimizer="sgd",
            momentum=0.9,
            weight_decay=1e-4,
            scheduler="cosine",
        )
    )

    @classmethod
    def b1_cars(cls) -> Exp7Config:
        """B1 headline: ImageNet-pretrained ResNet-50 full finetune on Stanford Cars.

        Production profile: training runs on the fast non-deterministic kernels; the
        measurements stay deterministic on their own (`deterministic_mode`). Methods are
        ordered with the K-FAC and COUPLE-FAC pair first, followed by the references
        and EKFAC family. The profile uses seeds 42-44 and the default refresh periods.
        """
        base = cls(
            methods=[
                "kfac",
                "kfac+overlay",
                "sgd",
                "newton_cg",
                "adam",
                "ekfac",
                "ekfac+overlay",
            ]
        )
        return replace(
            base,
            training=replace(base.training, seeds=[42, 43, 44], deterministic=False),
        )

    @classmethod
    def b1_mlp(cls) -> Exp7Config:
        """Optimizer-agnostic control on a full-rank (Linear) MLP - the clean EKFAC setting.

        The convolutional headline (`b1_cars`) has rank-deficient Kronecker factors, where
        EKFAC's eigenvalue correction is numerically delicate and close to K-FAC. This
        profile reruns the same optimizer comparison on a low-dimensional synthetic task
        whose Linear factors are full rank, so EKFAC is well conditioned and genuinely
        distinct - evidence that the overlay is agnostic to the block-diagonal base
        preconditioner, not specific to K-FAC.
        """
        return cls(
            dataset="synthetic",
            num_classes=10,
            model="mlp",
            pretrained=False,
            augment=False,
            # The label-noise ceiling is ~85%, and the from-scratch MLP has small initial
            # curvature, so the damped-Newton family needs a larger initial damping than
            # the pretrained-ResNet default (1e-2 diverges here) and a reachable target.
            damping=0.1,
            target_acc=0.7,
            # Fresh factors every step: the initial curvature is so small that factors
            # held for 20 steps blow the K-FAC step up (and the full-rank Linear factors
            # are cheap), so neither base preconditioner is amortized here.
            kfac_refresh_period=1,
            ekfac_refresh_period=1,
            training=TrainingConfig(
                lr=0.05,
                batch_size=256,
                epochs=20,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-4,
                scheduler="cosine",
            ),
        )


@dataclass
class Exp8Config:
    """Profiles for parameter-Hessian mass and block diagnostics.

    A1 compares ResNet-18 and its skip-free control at configured CIFAR-100
    checkpoints. A2 measures an ImageNet-pretrained ResNet-50 backbone with a
    new classifier on Stanford Cars without finetuning. Phi, diagonal norms
    and coupling are measured under the chosen partition and weight state;
    neither profile by itself identifies an architectural or pretraining cause.
    """

    # Backbones (each exposes group-major `get_param_groups`) + data
    archs: list[str] = field(default_factory=lambda: ["resnet18", "plain_resnet18"])
    dataset: str = "cifar100"  # "cifar100" | "cifar10" | "imagenet32" | "stanford_cars"
    num_classes: int = 100
    pretrained: bool = False  # ImageNet init (A2); False for from-scratch (A1)
    image_size: int = 32
    # DataLoader workers for the heavy decode paths (ImageNet-32 / Stanford Cars); raise
    # for the production ImageNet-32 anchor to clear the CPU-decode bottleneck. The light
    # CIFAR loaders keep their own small default.
    num_workers: int = 4

    # Phi estimation (Hutchinson on the parameter-space cross-block HVP)
    n_probes: int = 30
    hessian_batch_size: int = 32  # minibatch subsampled for the Hessian measurement

    # Checkpoints at which Phi is measured (init = pre-training, then mid, final)
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=0.1,
            batch_size=128,
            epochs=100,
            optimizer="sgd",
            momentum=0.9,
            weight_decay=5e-4,
            scheduler="cosine",
        )
    )

    # Where a checkpoint's state_dict is written when save_checkpoints is set
    # (build_config resolves this to the run's own --output-dir; a directly
    # constructed Exp8Config falls back to this literal). A run location, not a
    # protocol constant: excluded from the provenance block (experiments.provenance).
    output_dir: str = field(default="results/exp8", metadata={"provenance": False})
    # Write every measured checkpoint's state_dict to output_dir/checkpoints/
    # (experiments.protocol.save_checkpoints), so a later re-measurement restores a
    # weight snapshot instead of paying for another training run.
    save_checkpoints: bool = False
    # Read every requested checkpoint's state_dict back from this directory's own
    # checkpoints/ (experiments.protocol.load_checkpoints) instead of training: no
    # optimizer step runs, and Exp8Runner.run never builds a train loader either. The
    # directory is another run's own --output-dir from an earlier --save-checkpoints
    # run (same layout save_checkpoints wrote: <directory>/checkpoints/<arch>_seed
    # <seed>_ep<epoch>.pt). None (the default) means "train normally", which is why,
    # unlike output_dir, this field is NOT excluded from the provenance block: it
    # determines which weights a bundle was measured from, so a reader must be able
    # to tell a restored bundle apart from a freshly trained one instead of silently
    # mistaking one for the other.
    restore_checkpoints_dir: str | None = None
    # Additionally measure Phi at each of these probe counts, isolating Hutchinson
    # noise from seed-to-seed spread: every measure_phi call of the sweep shares one
    # generator seeded off the run's own seed, so the estimates differ by n_probes
    # alone and not by wherever the ambient global RNG happened to be. Empty (default)
    # adds nothing to the report.
    probe_sweep: list[int] = field(default_factory=list)
    # Adds phi_rank1 and lambda_max (one Lanczos run) to every measured checkpoint;
    # see experiments.phi.measure_phi and hessian.phi_baselines.phi_rank1.
    with_rank1: bool = False
    # Additionally measures Phi through experiments.regroup.Regrouped at each of these
    # levels (regroup.LEVELS) beyond the model's own partition ("stage", which
    # Regrouped defines as identical to it and which is therefore skipped): a finer
    # partition cannot discard less curvature mass, so Phi is non-decreasing from
    # stage to block to layer (not asserted here - a finite-probe estimate can dip
    # below its own noise floor on a given draw). Empty (default) adds nothing.
    granularity: list[str] = field(default_factory=list)

    @classmethod
    def a1_from_scratch(cls) -> Exp8Config:
        """A1 headline: ResNet-18 vs plain ResNet-18 from scratch on CIFAR-100.

        Production profile: training runs on the fast non-deterministic kernels; the Phi
        measurement stays deterministic on its own (`deterministic_mode`).
        """
        return cls._production(cls())

    @classmethod
    def a1_width(cls) -> Exp8Config:
        """ResNet-18 width sweep and ResNet-34 depth comparison on CIFAR-100.

        Uses the same schedule, seeds and checkpoints as a1_from_scratch.
        """
        return cls._production(
            cls(archs=["resnet18_w0.5", "resnet18", "resnet18_w2.0", "resnet34"])
        )

    @classmethod
    def a1_imagenet32(cls) -> Exp8Config:
        """ResNet-18 and skip-free comparison on downsampled ImageNet-32.

        The dataset mirror retains the index-0 background class, so the head
        has 1001 outputs. This optional profile uses a 30-epoch cosine schedule.
        """
        base = cls(dataset="imagenet32", num_classes=1001)
        return cls._production(
            replace(base, training=replace(base.training, epochs=30))
        )

    @staticmethod
    def _production(cfg: Exp8Config) -> Exp8Config:
        """Use nondeterministic training kernels; measurements set their own mode."""
        return replace(cfg, training=replace(cfg.training, deterministic=False))

    @classmethod
    def a2_pretrained(cls) -> Exp8Config:
        """Measure a pretrained ResNet-50 backbone with a new head on Stanford Cars.

        Runs only the initial checkpoint, without finetuning.
        """
        return cls(
            archs=["resnet50"],
            dataset="stanford_cars",
            num_classes=196,
            pretrained=True,
            image_size=224,
            checkpoint_epochs=["init"],
            training=TrainingConfig(
                lr=0.01,
                batch_size=64,
                epochs=0,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-4,
                scheduler="none",
            ),
        )


@dataclass
class Exp9Config:
    """Exp9: frozen-transformer Phi ladder.

    Measures parameter-Hessian Phi and block couplings for frozen decoder-only
    transformers across model sizes and weight states. The ``qkv`` attention
    partition exposes query, key and value parameter groups separately.

    Two backends share this schema:
      A3-toy (local, dependency-free) - `SegmentedTransformer` at growing sizes, measured
        on a random token minibatch; this is the default sweep and what the tests run; see
        `a3_toy`.
      A3-llm (on-demand) - frozen Hugging Face causal LMs (Qwen2.5-0.5B / 1.5B + a
        Llama-3.2-1B cross-family check), loaded in fp32 with gradients enabled on a 24 GB
        accelerator; see `a3_llm`.

    The model is built (toy) or loaded (hf) once per id under a fixed build seed; ``seeds``
    then vary only the Hutchinson probes, so their spread reports measurement precision.
    """

    backend: str = "toy"  # "toy" | "hf"
    # Toy size presets (see experiments.llm._TOY_PRESETS) or Hugging Face model ids.
    models: list[str] = field(default_factory=lambda: ["tiny", "small", "base"])
    seq_len: int = 64
    hessian_batch_size: int = 8  # minibatch subsampled for the Hessian measurement
    n_probes: int = 8  # Hutchinson probes per block (raised per publication)
    dtype: str = "float32"  # "float32" | "bfloat16"
    attn_granularity: str = "qkv"  # "qkv" (attention visible) | "block" (coarse)
    # Coupling-field scope: None measures every block pair (O(L^2); the toy ladder). An
    # int k keeps only pairs within k blocks in group order plus all intra-layer pairs, so
    # the on-demand LLM ladder stays O(L) without losing the attention hotspots.
    coupling_band: int | None = None
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])

    @classmethod
    def a3_toy(cls) -> Exp9Config:
        """A3-toy: local, dependency-free ladder over toy decoder sizes (tiny->small->base)."""
        return cls()

    @classmethod
    def a3_llm(cls) -> Exp9Config:
        """A3-llm: on-demand frozen-LLM ladder (Qwen2.5-0.5B / 1.5B + Llama-3.2-1B).

        Loaded in fp32 with gradients enabled for the parameter-space measurement, sized
        for a 24 GB accelerator: the full-parameter Hutchinson graph (weights + the
        create_graph gradient + the double backward) of the larger rungs (Qwen2.5-3B to
        32B, Llama-3.1-8B) does not fit in 24 GB fp32, so they are out of scope for this
        profile. 4-bit weights are out of scope (not differentiable). Requires the
        optional ``transformers`` dependency.
        """
        return cls(
            backend="hf",
            models=[
                "Qwen/Qwen2.5-0.5B",
                "meta-llama/Llama-3.2-1B",
            ],
            seq_len=128,
            hessian_batch_size=4,
            n_probes=8,
            dtype="float32",
            coupling_band=8,
            seeds=[42, 43, 44],
        )


def _exp10_mlp_training() -> TrainingConfig:
    """The Exp. 1 training protocol on the seeds 42-44 of the conditioning appendix."""
    return TrainingConfig(seeds=[42, 43, 44])


def _exp10_resnet_training() -> TrainingConfig:
    """The Exp. 6 training protocol on the single seed 42 of the conditioning appendix."""
    return TrainingConfig(lr=0.1, epochs=100, seeds=[42], weight_decay=5e-4)


@dataclass
class Exp10Config:
    """Exp10: conditioning of the blocks (appendix "Conditioning of the blocks").

    For a symmetric operator A - a diagonal block or the full matrix - the appendix
    reports the extreme eigenvalues lambda_max(A), lambda_min(A), the damped condition
    number kappa_lambda(A) = (lambda_max + lambda) / (lambda_min + lambda) (inf when
    lambda_min + lambda <= 0) for every lambda in ``lambdas``, the stable rank
    D(A) = ||A||_F^2 / ||A||_2^2 and, where the spectrum is exact, the fraction of
    negative eigenvalues (below -``neg_tol`` * max|lambda|). See `hessian.spectrum`.

    Two parts share the schema (``parts`` selects them):
      ``mlp``      - exact, activation space: the Exp. 1 Plain MLP (LayerNorm) and
        Residual MLP (alpha = L^{-1/2}) at ``depth`` x ``width`` on CIFAR-10, trained
        under the Exp. 1 protocol (``mlp_training``, seeds 42-44), the full block matrix
        [H^f_{v,w}] (depth * width square) assembled with `ExactBlockHessian` on the
        Exp. 1 Hessian minibatch at init and final, and its spectrum (and every diagonal
        block's) taken exactly.
      ``resnet18`` - matrix-free, parameter space: `SegmentedResNet18` on CIFAR-10 under
        the Exp. 6 protocol (``resnet_training``, seed 42); the six diagonal parameter
        blocks H_{theta_v, theta_v} and the full H_theta are accessed through the exact
        HVP only - lambda_max and lambda_min together from one ``n_lanczos_iter``-step
        Lanczos run (`hessian.spectrum.extreme_ritz`; the three power iterations this
        replaces could not resolve lambda_max when the dominant eigenvalue is the
        negative one) and ||A||_F^2 by ``n_probes`` Hutchinson probes - on a fixed
        validation minibatch at init and final.
    """

    parts: list[str] = field(default_factory=lambda: ["mlp", "resnet18"])
    lambdas: list[float] = field(default_factory=lambda: [1e-2, 1e-1])
    n_lanczos_iter: int = 60  # Lanczos steps for both extremes (matrix-free part)
    n_probes: int = 30  # Hutchinson probes for ||A||_F^2 (matrix-free part)
    neg_tol: float = 1e-8  # negative-eigenvalue threshold, relative to max|lambda|
    hessian_batch_size: int = 32  # the Exp. 1 / Exp. 6 Hessian minibatch
    checkpoint_epochs: list[str] = field(default_factory=lambda: ["init", "final"])

    # Exact part: the Exp. 1 MLPs (Plain with LayerNorm, Residual with alpha = L^{-1/2}).
    depth: int = 8
    width: int = 64
    use_layernorm: bool = True
    mlp_training: TrainingConfig = field(default_factory=_exp10_mlp_training)

    # Matrix-free part: ResNet-18 under the Exp. 6 training protocol.
    resnet_training: TrainingConfig = field(default_factory=_exp10_resnet_training)


def _exp11_protocol(base: Exp7Config) -> Exp7Config:
    """An Exp. 7 profile narrowed to the single seed 42 - the K-FAC run exp11 probes.

    Everything else (backbone, data, lambda_0, the Fisher estimate, the factor refresh
    period T_k, the batch, the fast kernels) is the profile's; ``--seeds`` adds seeds.
    """
    return replace(base, training=replace(base.training, seeds=[42]))


@dataclass
class Exp11Config:
    """Exp11: model validity radius along an optimization trajectory.

    The runner compares loss changes along the selected K-FAC, SGD or Adam trajectory
    with the damped quadratic model m(t d0) = t g^T d0 +
    t^2/2 d0^T (H + lambda I) d0. Model/loss disagreement includes the explicit
    damping penalty and therefore is not solely undamped curvature error. The K-FAC
    protocol uses fixed damping. At every step in ``probe_steps`` (the run trains up to the
    last one), on that step's minibatch and before the step is applied, it records

      (a) the decomposition of m(d0): g^T d0, d0^T H d0, lambda ||d0||^2, ||d0||, ||g||,
          eta_eff = ||d0|| / ||g|| and eta_g = -g^T d0 / ||g||^2;
      (b) the curve L(theta + t d0) - L(theta) against m(t d0) for t in ``t_grid``, the
          secant curvature 2 (Delta L - t g^T d0) / t^2 at every t, and the validity
          radius t* - the largest grid t up to which |Delta L - m| <= ``tolerance``
          |Delta L| holds at every grid point;
      (c) the same curve along the top Hessian eigenvector scaled to ||d0|| (t on the
          same footing), oriented downhill - the range of the sharp directions;
      (d) lambda_max and lambda_min by one ``n_lanczos_iter``-step Lanczos run
          (`hessian.spectrum.extreme_ritz`, floored by the Rayleigh quotients along d0
          and g) and the curvature along g, with lambda_max * eta_eff, the quadratic
          edge-of-stability number;
      (e) the kinks the step crosses: ReLU pre-activation sign flips and max-pool
          argmax changes between theta and theta + t d0 for t in ``kink_t_grid`` - the
          non-smoothness a pointwise Hessian cannot see;

    plus the Exp. 7 payoff numbers on the same objects (m(d0), the losses before and
    after the step, rho), so the two experiments cross-check, and the minibatch loss
    before every step of the trajectory.
    """

    protocol: Exp7Config = field(
        default_factory=lambda: _exp11_protocol(Exp7Config.b1_cars())
    )
    # Steps probed (before the step is applied); training stops after the last one.
    probe_steps: list[int] = field(default_factory=lambda: [0, 20, 60, 200, 600, 1500])
    # Fractions t of the step at which the loss curve and the model are compared.
    t_grid: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    # Fractions t at which the kink crossings are counted (a forward each).
    kink_t_grid: list[float] = field(default_factory=lambda: [1e-3, 1e-2, 0.1, 1.0])
    # Relative model error |Delta L - m| / |Delta L| the validity radius allows.
    tolerance: float = 0.1
    # Lanczos steps for both extremes; the top Ritz vector doubles it in HVPs.
    n_lanczos_iter: int = 60
    # Which trajectory's step the probes measure; a key of
    # `experiments.runner_exp11.STEP_SOURCES` (``--step-source``).
    step_source: str = "kfac"

    @classmethod
    def b1_cars(cls) -> Exp11Config:
        """The headline: the b1 K-FAC run of Exp. 7 (ResNet-50 on Stanford Cars)."""
        return cls()

    @classmethod
    def b1_mlp(cls) -> Exp11Config:
        """The cheap control on the synthetic full-rank MLP of `Exp7Config.b1_mlp`."""
        return cls(protocol=_exp11_protocol(Exp7Config.b1_mlp()))
