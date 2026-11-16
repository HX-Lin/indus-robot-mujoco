"""Configuration objects for the JAX VITA ODE + predicted-action PPF port.

The defaults mirror the active ``vita_ode_ppf_pred_action_10k`` launch:
absolute dual-arm 14-D ODE action head, FASA object-change v2, Flan-T5-small
clause/object/reference conditioning, grounded stage routing, and the
training-only t+4 future-visual PPF auxiliary.  Paths are deliberately not
embedded here; they are supplied by the caller or the entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


def _rgb(values):
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class ModelConfig:
    # Observation / cameras
    image_keys: Tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    obs_horizon: int = 2
    resize_shape: Tuple[int, int] = (240, 320)
    crop_shape: Tuple[int, int] = (224, 308)
    photometric_aug: bool = True
    input_norm: bool = True
    robust_aug: bool = False

    # Visual backbone
    backbone: str = "resnet18"
    # Local torchvision-compatible ResNet18 ImageNet state_dict.  Fresh
    # production training requires this path; synthetic tests may opt out.
    resnet18_imagenet_weights: str = ""
    backbone_trainable: bool = True
    # Frozen BatchNorm statistics and affine values are stored in the
    # non-optimized ``constants`` collection; convolutions remain trainable.
    backbone_frozen_bn: bool = True

    # FASA
    spatial_mode: str = "fasa"
    use_multiscale: bool = True
    use_activation_prior: bool = True
    fasa_cross_camera: bool = True
    fasa_camera_role_embed: bool = True
    fasa_camera_scale: bool = True
    fasa_cosine_change: bool = True
    fasa_object_change: bool = True
    fasa_object_change_v2: bool = True
    fasa_core_ring: bool = False
    fasa_source_null: bool = False
    fasa_per_cam_dim: int = 1060
    fasa_coord_dim: int = 32
    fasa_router_input_dim: int = 516
    trajectory_spatial_grid: Tuple[int, int] = (7, 10)

    # Language
    language_hidden_dim: int = 256
    language_max_length: int = 16
    language_order: str = "red green blue"
    generated_text_targets: bool = True
    joint_planner: bool = True
    target_descriptions: Optional[Tuple[str, ...]] = None
    reference_texts: Optional[Tuple[str, ...]] = None
    language_spatial_gate: bool = True
    target_grounding: bool = True
    n_clauses: int = 3

    # State / action
    state_dim: int = 16
    action_dim: int = 14
    chunk_size: int = 16
    action_horizon: int = 8
    state_dropout: float = 0.15
    state_delta_dropout: float = 0.0

    # FASA object-change static-context dropout (active run: 0)
    static_context_dropout: float = 0.0

    # ODE
    ode_solver: str = "euler"
    ode_hidden_dim: int = 256
    ode_num_layers: int = 4
    ode_perturb_prob: float = 0.5
    ode_perturb_std: float = 0.02
    ode_spatial_attention: bool = False
    ode_relation_readout: bool = False
    spatial_action_memory: bool = True
    spatial_full_context: bool = True

    # PPF future-visual auxiliary
    ppf_future_visual_weight: float = 0.1
    ppf_future_visual_horizon: int = 4
    ppf_future_visual_hidden: int = 512

    # Objective weights
    flow_recon_weight: float = 0.5
    target_grounding_weight: float = 0.5
    stage_route_weight: float = 0.5
    joint_plan_weight: float = 0.5
    # The active run does not enable the legacy PPF-lite progress/dynamics
    # heads (no --use_ppf); they are intentionally absent here.
    use_ppf_lite: bool = False

    # Stage route labels come from the sidecar; the router has n_clauses bins.
    stage_router_hidden: int = 128

    # Deterministic evaluation crop
    eval_center_crop: bool = True

    def validate(self):
        if self.spatial_mode != "fasa":
            raise ValueError("the active configuration requires spatial_mode='fasa'")
        if self.fasa_object_change and not self.fasa_object_change_v2:
            raise ValueError("fasa_object_change requires fasa_object_change_v2")
        if self.fasa_source_null and not self.fasa_object_change_v2:
            raise ValueError("fasa_source_null requires fasa_object_change_v2")
        if self.fasa_source_null and self.fasa_core_ring:
            raise ValueError("fasa_source_null is incompatible with fasa_core_ring")
        if self.ode_spatial_attention and not self.spatial_action_memory:
            raise ValueError("ODE spatial attention requires spatial_action_memory")
        if self.state_dim != 16 or self.action_dim != 14:
            raise ValueError("ODE absolute pose path requires 16-D state and 14-D action")
        if self.chunk_size < self.ppf_future_visual_horizon:
            raise ValueError("chunk_size must cover the PPF future-visual horizon")
        if self.crop_shape[0] > self.resize_shape[0] or self.crop_shape[1] > self.resize_shape[1]:
            raise ValueError("crop_shape must fit inside resize_shape")
        if self.ppf_future_visual_weight < 0:
            raise ValueError("ppf_future_visual_weight must be non-negative")
        if self.static_context_dropout != 0.0:
            raise NotImplementedError(
                "static_context_dropout is fixed at the active run's 0.0; "
                "the JAX port does not implement the optional dropout")
        if self.robust_aug:
            raise NotImplementedError(
                "robust_aug is disabled in the active run and is not ported")
        if self.use_ppf_lite:
            raise NotImplementedError(
                "legacy PPF-lite heads are disabled in the active run")
        return self


@dataclass(frozen=True)
class DataConfig:
    # All paths are runtime parameters.  Do not hard-code user locations.
    data_dir: str = ""
    repo_id: str = ""
    ee_cache: str = ""
    stage_cache: str = ""
    grounding_cache: str = ""
    video_backend: str = "torchcodec"
    # The active PyTorch run has val_freq=0, so all 50 episodes train.
    val_split: float = 0.0
    seed: int = 42
    # Number of trailing frames removed from every episode.  The active run
    # uses chunk_size - 1; the future t+4 mask is still computed independently.
    drop_n_last_frames: Optional[int] = None
    future_visual_horizon: int = 4

    def validate(self):
        if self.val_split != 0.0:
            raise NotImplementedError(
                "the active JAX port currently mirrors the run with val_freq=0; "
                "episode-held-out validation is not enabled")
        for name in ("data_dir", "repo_id", "ee_cache"):
            if not getattr(self, name):
                raise ValueError(f"DataConfig.{name} must be set by the caller")
        if self.future_visual_horizon < 0:
            raise ValueError("future_visual_horizon must be non-negative")
        return self


@dataclass(frozen=True)
class TextConfig:
    # 't5_flax' uses transformers' Flax T5 seq2seq model on a local
    # Flan-T5-small checkpoint.  The same encoder feeds FASA while the decoder
    # receives the teacher-forced planner loss.  'synthetic' is a deterministic
    # test-only fallback; 'none' raises if the policy needs text conditioning.
    encoder: str = "t5_flax"
    model_path: str = ""
    max_length: int = 16
    order: str = "red green blue"
    generated_text_targets: bool = True
    joint_planner: bool = True
    joint_plan_batch_size: int = 8
    planner_max_length: int = 192
    planner_labels_max_length: int = 128
    target_descriptions: Optional[Tuple[str, ...]] = None
    reference_texts: Optional[Tuple[str, ...]] = None
    task_instruction: Optional[str] = None

    def validate(self):
        if self.encoder == "t5_flax" and not self.model_path:
            raise ValueError("TextConfig.model_path is required for t5_flax")
        if self.generated_text_targets and not self.joint_planner:
            raise ValueError(
                "generated_text_targets requires the joint T5 planner")
        if (self.encoder == "t5_flax" and self.joint_planner
                and not self.task_instruction):
            raise ValueError(
                "the joint T5 planner requires TextConfig.task_instruction")
        if self.joint_plan_batch_size < 0:
            raise ValueError("joint_plan_batch_size must be non-negative")
        if self.planner_max_length < 1 or self.planner_labels_max_length < 1:
            raise ValueError("planner token lengths must be positive")
        return self


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10000
    batch_size: int = 16
    lr: float = 1e-4
    language_lr: float = 1e-5
    weight_decay: float = 1e-6
    beta1: float = 0.95
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 10.0
    warmup_steps: int = 500
    seed: int = 42
    save_freq: int = 10000
    eval_freq: int = 0
    output_dir: str = "vita_jax_runs"
    resume: Optional[str] = None
    log_freq: int = 100
    # Set True only when the caller explicitly wants to evaluate the active
    # objective; future-visual remains diagnostic-only in eval either way.
    eval_batches: int = 0
    jit: bool = True


@dataclass(frozen=True)
class VITAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        self.model.validate()
        self.data.validate()
        self.text.validate()
        if self.train.steps < 0 or self.train.batch_size <= 0:
            raise ValueError("invalid training steps/batch size")
        return self
