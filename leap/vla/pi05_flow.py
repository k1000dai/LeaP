"""Flow reversal and noise injection on LeRobot's ``PI05Policy`` (PyTorch port of openpi pi0.5).

Everything here wraps the user's LeRobot clone (0.6.1, commit 9459727e7) without changing it.
LeRobot is imported lazily inside the functions that need it, so the integrator math
(``forward_from_noise``, ``reverse_actions``) can be tested with a fake policy on CPU.

Flow convention (``lerobot/policies/common/flow_matching.py`` and ``pi05/modeling_pi05.py``,
same as openpi):

    x_t = t * noise + (1 - t) * actions,     u_t = noise - actions     (t in [0, 1])

``denoise_step`` predicts ``u_t`` from ``x_t`` and ``t``.  Sampling integrates from t = 1
(noise) down to t = 0 (actions) with forward Euler:

    dt = -1 / K,  for step in 0..K-1:  t = 1 + step * dt,  x <- x + dt * v(x, t)

Flow Reversal Steering (Tang et al. 2026, App. B) runs the same field the other way, from
the (normalised, zero-padded) action chunk at t = 0 up to the noise at t = 1:

    h = 1 / K,  for step in 0..K-1:   t = step * h,        x <- x + h * v(x, t)

Moving along +u with increasing t goes from actions towards noise, which is why the
reversal adds ``h * v`` while the sampler subtracts it.  K = 10 (h = 0.1) in both directions
matches the checkpoint's ``num_inference_steps`` and the FRS paper.

Tensor contracts (B = batch, chunk_size = 50, max_action_dim = 32, action_dim = 7):

- noise / x_t / velocities: float32 ``[B, 50, 32]`` on the policy device; dims >= 7 are
  padding (zero in actions, i.i.d. N(0, I) in noise, see ``pad_noise``).
- un-normalised env actions: ``[B, 50, 7]``.
- prefix features: float32 ``[B, FEAT_DIM]`` = mean-pooled projected image tokens of the two
  real cameras (2 x 2048, ``PaliGemmaWithExpertModel.embed_image``) + the normalised 8-d
  state as the preprocessor hands it to the policy.

Normalisation: the checkpoint ``lerobot/pi05_libero_finetuned_v044`` uses MEAN_STD for
``action`` and ``observation.state`` (``normalization_mapping`` in its config.json).  The
statistics live in the ``NormalizerProcessorStep`` of the pre-processor pipeline
(``policy_preprocessor_step_2_normalizer_processor.safetensors``) and, mirrored, in the
``UnnormalizerProcessorStep`` of the post-processor; ``normalize_actions`` /
``unnormalize_actions`` reuse those steps' ``stats["action"]`` and apply exactly their
formulas: ``(a - mean) / (std + eps)`` forward and ``x * std + mean`` inverse.

Batch construction mirrors ``lerobot/scripts/lerobot_eval.py``: ``preprocess_observation``
turns uint8 HxWx3 renders into float32 [B, 3, H, W] in [0, 1]; the LIBERO env processor
(``LiberoProcessorStep``) rotates the images by 180 degrees (flip H and W) and builds the
8-d state (eef pos, eef axis-angle, gripper qpos); ``observation["task"]`` carries the
prompt; then the policy pre-processor normalises the state, discretises it into the prompt,
tokenises (PaliGemma tokenizer, padded to ``tokenizer_max_length`` = 200) and moves to the
device.  ``prepare_batch`` takes the raw renders and the 8-d state and applies the flip
itself (``flip_images=True``), so pass raw LIBERO renders, not already-rotated ones.
"""
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
IMAGE_KEY = "observation.images.image"
IMAGE2_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
TOKENS_KEY = "observation.language.tokens"
MASK_KEY = "observation.language.attention_mask"
ACTION_KEY = "action"
TASK_KEY = "task"

IMAGE_TOKEN_DIM = 2048  # PaliGemma multi_modal_projector output (vision_config.projection_dim)
NUM_REAL_CAMERAS = 2  # agentview + wrist; the third camera of the checkpoint is empty
STATE_DIM = 8
ACTION_DIM = 7
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
FEAT_DIM = NUM_REAL_CAMERAS * IMAGE_TOKEN_DIM + STATE_DIM  # 4104
DEFAULT_NUM_STEPS = 10
DEFAULT_EPS = 1e-8  # NormalizerProcessorStep.eps

# Accepted names for the two real cameras in ``prepare_batch(images=...)``.
_CAMERA_ALIASES = {
    IMAGE_KEY: ("image", "agentview", "agentview_image", IMAGE_KEY),
    IMAGE2_KEY: ("image2", "wrist", "robot0_eye_in_hand_image", IMAGE2_KEY),
}


@dataclass
class Pi05Bundle:
    """A loaded policy with the pre/post-processors ``lerobot-eval`` would build for it."""

    policy: Any
    preprocessor: Any
    postprocessor: Any
    config: Any
    device: torch.device


@dataclass
class PrefixContext:
    """Cached prefix (images + language) of one batch, as ``sample_actions`` computes it.

    ``past_key_values`` is the PaliGemma KV cache of the prefix; ``denoise_step`` clones it
    on every call (``modeling_pi05.py``), so one context serves any number of velocity
    evaluations.  ``prefix_embs`` and ``image_token_counts`` let ``prefix_features`` reuse
    the image tokens without a second SigLIP pass.
    """

    prefix_pad_masks: Tensor
    past_key_values: Any
    images: list = field(default_factory=list)
    img_masks: list = field(default_factory=list)
    tokens: Tensor | None = None
    masks: Tensor | None = None
    prefix_embs: Tensor | None = None
    image_token_counts: list = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.prefix_pad_masks.shape[0])


@dataclass
class Observation:
    """One raw observation: uint8 HxWx3 renders per camera, the 8-d state and the prompt."""

    images: dict
    state: np.ndarray
    prompt: str


# --------------------------------------------------------------------------- loading


def load_pi05(checkpoint=CHECKPOINT, device="cuda", n_action_steps=10, dtype=None,
              compile_model=False) -> Pi05Bundle:
    """Load ``PI05Policy`` and its processors the way ``lerobot-eval`` does.

    The config comes from the checkpoint; ``device``, ``n_action_steps`` (10, as the
    sibling ``lerobot/pi05-libero`` sets) and optionally ``dtype`` ("bfloat16" /
    "float32") override it.  ``compile_model`` defaults to False: the checkpoint asks for
    ``torch.compile(max-autotune)`` of ``sample_actions``, which this module never calls
    (it drives ``denoise_step`` directly), so compiling would only add warm-up time.
    The pre-processor's device step is overridden to the policy device and the rename map
    is empty, exactly as ``lerobot_eval.main`` does.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = str(device)
    config.n_action_steps = n_action_steps
    config.compile_model = compile_model
    config.gradient_checkpointing = False  # inference only; the flag is a no-op in eval mode anyway
    if dtype is not None:
        config.dtype = dtype
    policy = PI05Policy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()
    policy.reset()  # the action queue length follows n_action_steps
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides=preprocessor_overrides)
    return Pi05Bundle(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                      config=policy.config, device=torch.device(device))


# --------------------------------------------------------------------------- batches


def _camera_image(images: dict, key: str) -> np.ndarray:
    for alias in _CAMERA_ALIASES[key]:
        if alias in images:
            return images[alias]
    raise KeyError(f"No image for {key}; expected one of {_CAMERA_ALIASES[key]} in {sorted(images)}")


def image_to_tensor(image: np.ndarray, flip: bool = True) -> Tensor:
    """uint8 HxWx3 -> float32 3xHxW in [0, 1], rotated by 180 degrees when ``flip``.

    Mirrors ``lerobot.envs.utils.preprocess_observation`` (channel-last uint8 to
    channel-first float / 255) followed by ``LiberoProcessorStep`` (``torch.flip`` over H
    and W, the HuggingFaceVLA/libero camera convention).
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a uint8 HxWx3 image, got {image.dtype} {image.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    if flip:
        tensor = torch.flip(tensor, dims=[1, 2])
    return tensor


def raw_batch(observations: Iterable[Observation], flip_images: bool = True) -> dict:
    """Stack raw observations into the un-processed batch ``lerobot-eval`` hands to the
    policy pre-processor (images [B, 3, H, W] in [0, 1], state [B, 8], task list)."""
    observations = list(observations)
    if not observations:
        raise ValueError("No observations")
    batch = {}
    for key in (IMAGE_KEY, IMAGE2_KEY):
        batch[key] = torch.stack([image_to_tensor(_camera_image(obs.images, key), flip_images)
                                  for obs in observations])
    states = [np.asarray(obs.state, dtype=np.float32).reshape(-1) for obs in observations]
    if any(state.shape[0] != STATE_DIM for state in states):
        raise ValueError(f"Expected {STATE_DIM}-d states, got {[s.shape for s in states]}")
    batch[STATE_KEY] = torch.from_numpy(np.stack(states))
    batch[TASK_KEY] = [str(obs.prompt) for obs in observations]
    return batch


def prepare_batches(bundle: Pi05Bundle, observations: Iterable[Observation],
                    flip_images: bool = True) -> dict:
    """Pre-process a list of observations into one policy batch (tokens are padded to the
    tokenizer max length, so stacking is exact)."""
    return bundle.preprocessor(raw_batch(observations, flip_images))


def prepare_batch(bundle: Pi05Bundle, images: dict, state: np.ndarray, prompt: str,
                  flip_images: bool = True) -> dict:
    """Policy input batch for ONE observation (``observation.images.image``,
    ``observation.images.image2``, ``observation.state``, ``task`` -> pre-processor)."""
    return prepare_batches(bundle, [Observation(images, state, prompt)], flip_images)


# --------------------------------------------------------------------------- prefix


@torch.no_grad()
def prefix_context(bundle: Pi05Bundle, batch: dict) -> PrefixContext:
    """Embed the prefix and run PaliGemma once with ``use_cache=True`` (as
    ``PI05FlowMatching.sample_actions`` does before its Euler loop)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

    policy = bundle.policy
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[TOKENS_KEY], batch[MASK_KEY]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    total_image_tokens = prefix_embs.shape[1] - tokens.shape[1]
    per_camera = total_image_tokens // len(images)
    return PrefixContext(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                         images=list(images), img_masks=list(img_masks), tokens=tokens, masks=masks,
                         prefix_embs=prefix_embs, image_token_counts=[per_camera] * len(images))


@torch.no_grad()
def prefix_features(bundle: Pi05Bundle, batch: dict, ctx: PrefixContext | None = None) -> Tensor:
    """Observation features for the prior network: ``[B, FEAT_DIM]`` float32.

    Mean over the projected image tokens of each real camera (empty cameras, whose
    ``img_masks`` are all False, are skipped) concatenated with the normalised state from
    the pre-processed batch.  With ``ctx`` the tokens are read from the cached prefix
    embeddings instead of re-running ``embed_image``.
    """
    policy = bundle.policy
    if ctx is not None and ctx.prefix_embs is not None:
        images, img_masks = ctx.images, ctx.img_masks
        pooled, offset = [], 0
        for img_mask, count in zip(img_masks, ctx.image_token_counts):
            if bool(img_mask.any()):
                pooled.append(ctx.prefix_embs[:, offset:offset + count].float().mean(dim=1))
            offset += count
    else:
        images, img_masks = policy._preprocess_images(batch)
        pooled = []
        for img, img_mask in zip(images, img_masks):
            if bool(img_mask.any()):
                tokens = policy.model.paligemma_with_expert.embed_image(img)
                pooled.append(tokens.float().mean(dim=1))
    if len(pooled) != NUM_REAL_CAMERAS:
        raise ValueError(f"Expected {NUM_REAL_CAMERAS} real cameras, found {len(pooled)}")
    state = batch[STATE_KEY][:, :STATE_DIM].float()
    feat = torch.cat(pooled + [state.to(pooled[0].device)], dim=1)
    if feat.shape[1] != FEAT_DIM:
        raise ValueError(f"Feature dim {feat.shape[1]} != FEAT_DIM {FEAT_DIM}")
    return feat


# --------------------------------------------------------------------------- flow


def _time_tensor(t, x_t: Tensor) -> Tensor:
    """Float32 ``[B]`` timestep tensor on ``x_t``'s device, from a scalar or a tensor."""
    bsize = x_t.shape[0]
    if isinstance(t, Tensor):
        t = t.to(dtype=torch.float32, device=x_t.device)
        return t.expand(bsize) if t.dim() == 0 else t.reshape(bsize)
    return torch.tensor(float(t), dtype=torch.float32, device=x_t.device).expand(bsize)


@torch.no_grad()
def velocity(bundle: Pi05Bundle, ctx: PrefixContext, x_t: Tensor, t) -> Tensor:
    """``u_t = denoise_step(x_t, t)`` with the cached prefix (the cache is cloned inside)."""
    return bundle.policy.model.denoise_step(
        prefix_pad_masks=ctx.prefix_pad_masks,
        past_key_values=ctx.past_key_values,
        x_t=x_t,
        timestep=_time_tensor(t, x_t),
    )


@torch.no_grad()
def forward_from_noise(bundle: Pi05Bundle, ctx: PrefixContext, noise: Tensor,
                       num_steps: int = DEFAULT_NUM_STEPS) -> Tensor:
    """Standard sampler: t from 1 to 0, ``x <- x + dt * v``, ``dt = -1/K``.

    Reproduces ``euler_integrate`` (and thus ``sample_actions(noise=noise)``) step by step,
    including the float arithmetic of the timestep.
    """
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        v_t = velocity(bundle, ctx, x_t, time)
        x_t = x_t + dt * v_t
    return x_t


@torch.no_grad()
def reverse_actions(bundle: Pi05Bundle, ctx: PrefixContext, actions_norm_padded: Tensor,
                    num_steps: int = DEFAULT_NUM_STEPS) -> Tensor:
    """Flow reversal: from the normalised, zero-padded chunk ``[B, 50, 32]`` at t = 0 up to
    the noise at t = 1 with ``x <- x + (1/K) * v(x, t)`` for t = 0, 1/K, ..., (K-1)/K."""
    h = 1.0 / num_steps
    x_t = actions_norm_padded
    for step in range(num_steps):
        time = step * h
        v_t = velocity(bundle, ctx, x_t, time)
        x_t = x_t + h * v_t
    return x_t


# --------------------------------------------------------------------------- normalisation


def _find_stats_step(pipeline):
    for step in getattr(pipeline, "steps", []):
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and ACTION_KEY in stats:
            return step
    return None


def action_stats(bundle: Pi05Bundle) -> tuple[Tensor, Tensor, float]:
    """``(mean, std, eps)`` of the ``action`` MEAN_STD statistics, as float32 ``[7]`` tensors
    on the bundle device, read from the normaliser step of the pre-processor (falling back
    to the un-normaliser of the post-processor)."""
    step = _find_stats_step(bundle.preprocessor) or _find_stats_step(bundle.postprocessor)
    if step is None:
        raise ValueError("No processor step with action normalisation stats found")
    stats = step.stats[ACTION_KEY]
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"MEAN_STD stats need mean and std, got {sorted(stats)}")
    mean = torch.as_tensor(np.asarray(stats["mean"]), dtype=torch.float32, device=bundle.device).reshape(-1)
    std = torch.as_tensor(np.asarray(stats["std"]), dtype=torch.float32, device=bundle.device).reshape(-1)
    return mean, std, float(getattr(step, "eps", DEFAULT_EPS))


def action_dim(bundle: Pi05Bundle) -> int:
    return int(bundle.config.output_features[ACTION_KEY].shape[0])


def pad_actions(actions: Tensor, max_action_dim: int = MAX_ACTION_DIM) -> Tensor:
    """Zero-pad the last dim to ``max_action_dim`` (``vla_utils.pad_vector``)."""
    if actions.shape[-1] >= max_action_dim:
        return actions
    pad = torch.zeros(*actions.shape[:-1], max_action_dim - actions.shape[-1],
                      dtype=actions.dtype, device=actions.device)
    return torch.cat([actions, pad], dim=-1)


def normalize_actions(bundle: Pi05Bundle, actions7) -> Tensor:
    """``[B, 50, 7]`` env actions -> ``[B, 50, 32]`` normalised, zero-padded, float32.

    Same transform as the pre-processor applies to ``action`` during training:
    ``(a - mean) / (std + eps)`` (``NormalizerProcessorStep``), then ``pad_vector``.
    """
    mean, std, eps = action_stats(bundle)
    actions = torch.as_tensor(np.asarray(actions7) if not isinstance(actions7, Tensor) else actions7,
                              dtype=torch.float32).to(bundle.device)
    if actions.shape[-1] != mean.shape[0]:
        raise ValueError(f"Expected {mean.shape[0]}-d actions, got {actions.shape}")
    normalised = (actions - mean) / (std + eps)
    return pad_actions(normalised, int(bundle.config.max_action_dim))


def unnormalize_actions(bundle: Pi05Bundle, actions_norm) -> Tensor:
    """``[B, 50, 32]`` normalised -> ``[B, 50, 7]`` env actions (float32), mirroring
    ``predict_action_chunk`` (drop padding dims) and the post-processor
    (``UnnormalizerProcessorStep``: ``x * std + mean``)."""
    mean, std, _ = action_stats(bundle)
    actions = torch.as_tensor(actions_norm, dtype=torch.float32).to(bundle.device)
    return actions[..., :action_dim(bundle)] * std + mean


# --------------------------------------------------------------------------- sampling


def sample_noise(shape, device) -> Tensor:
    """Standard-normal float32 noise exactly as ``flow_matching.sample_noise`` draws it."""
    return torch.normal(mean=0.0, std=1.0, size=tuple(shape), dtype=torch.float32, device=device)


def pad_noise(noise7, max_action_dim: int = MAX_ACTION_DIM, generator=None) -> Tensor:
    """``[B, 50, 7]`` source noise -> ``[B, 50, 32]`` with dims 7..31 drawn i.i.d. N(0, I).

    FRS replaces the padding dims of the reversed noise by fresh Gaussian noise (no effect
    on performance); the prior network predicts only the first ``action_dim`` dims.
    """
    noise = torch.as_tensor(np.asarray(noise7) if not isinstance(noise7, Tensor) else noise7,
                            dtype=torch.float32)
    extra = max_action_dim - noise.shape[-1]
    if extra < 0:
        raise ValueError(f"Noise has {noise.shape[-1]} dims > max_action_dim {max_action_dim}")
    if extra == 0:
        return noise
    pad = torch.randn(*noise.shape[:-1], extra, dtype=torch.float32, device=noise.device,
                      generator=generator)
    return torch.cat([noise, pad], dim=-1)


@torch.no_grad()
def sample_actions_with_noise(bundle: Pi05Bundle, ctx: PrefixContext, noise=None,
                              num_steps: int = DEFAULT_NUM_STEPS) -> np.ndarray:
    """Denoise ``noise`` ``[B, 50, 32]`` (Gaussian when None) with the cached prefix and
    return un-normalised env actions ``[B, 50, 7]`` as float32 numpy."""
    if noise is None:
        shape = (ctx.batch_size, int(bundle.config.chunk_size), int(bundle.config.max_action_dim))
        noise = sample_noise(shape, bundle.device)
    noise = torch.as_tensor(noise, dtype=torch.float32).to(bundle.device)
    actions_norm = forward_from_noise(bundle, ctx, noise, num_steps)
    return unnormalize_actions(bundle, actions_norm).cpu().numpy().astype(np.float32)
