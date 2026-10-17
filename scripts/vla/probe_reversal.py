#!/usr/bin/env python
"""Flow-reversal round-trip probe for LeRobot's pi0.5 LIBERO checkpoint.

Standalone diagnostic that uses only LeRobot APIs. It loads the policy exactly like
`lerobot-eval` does, builds one real observation batch from a LIBERO env, then measures:

  (a) a = sample_actions(noise=z, 10 steps) for z ~ N(0, I) of shape [1, 50, 32]
  (b) reversal: x <- a; for t in 0, 1/K, ..., (K-1)/K: x <- x + (1/K) * v(x, t); z_hat = x
  (c) ||z_hat - z|| / ||z|| over the 7 real action dims and over the 25 padding dims,
      and the action round trip ||sample_actions(noise=z_hat) - a|| / ||a||
  (d) the same with K = 20 (and 50) for the reversal while the forward pass keeps 10 steps
  (e) a shifted reference a' = a with +0.3 added to the normalised dx dim of the first 10
      steps, reversed and re-sampled: how much of the shift survives the round trip
  timings for sample_actions at batch 1 and 16 and for the reversal, and GPU memory.

Run on the remote inside the VLA venv, e.g.

  cd ~/codes/LeaP && MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/data/vla/libero_config \
  CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
  .venv-vla/bin/python scripts/vla/probe_reversal.py --dtype bfloat16 \
      --out logs/vla/probe_reversal_bf16.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

CKPT = "lerobot/pi05_libero_finetuned_v044"
REAL = slice(0, 7)  # LIBERO action dims inside the 32-wide padded chunk
PAD = slice(7, 32)


# ----------------------------------------------------------------------------- loading
def load_policy(ckpt: str, dtype: str, n_action_steps: int, compile_model: bool, env_cfg):
    """Mirror lerobot_eval.eval_main: config from the Hub, overrides, make_policy, processors."""
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy_cfg.pretrained_path = ckpt  # make_policy only loads weights when this is set
    policy_cfg.n_action_steps = n_action_steps
    policy_cfg.dtype = dtype
    policy_cfg.compile_model = compile_model
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={
            "device_processor": {"device": str(policy_cfg.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return policy, preprocessor, postprocessor, env_pre, env_post


def make_batch(env_cfg, preprocessor, env_pre, seed: int):
    """One real observation from the LIBERO env after reset, run through the same pipeline as rollout()."""
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = next(iter(next(iter(envs.values())).values()))
    raw_obs, _info = vec_env.reset(seed=[seed])
    task = list(vec_env.call("task_description"))
    observation = preprocess_observation(raw_obs)
    observation["task"] = task
    observation = env_pre(observation)
    batch = preprocessor(observation)
    return vec_env, batch, task[0]


# ---------------------------------------------------------------------- flow pieces
@torch.no_grad()
def model_inputs(policy, batch):
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    return images, img_masks, tokens, masks


@torch.no_grad()
def prefix_cache(model, images, img_masks, tokens, masks):
    """Copy of the prefix/KV-cache lines of PI05FlowMatching.sample_actions."""
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
    return prefix_pad_masks, past_key_values


@torch.no_grad()
def velocity(model, prefix_pad_masks, past_key_values, x, t: float):
    t_tensor = torch.full((x.shape[0],), t, dtype=torch.float32, device=x.device)
    return model.denoise_step(
        prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x, timestep=t_tensor
    )


@torch.no_grad()
def reverse(model, prefix_pad_masks, past_key_values, actions, num_steps: int):
    """Actions (t=0) -> noise (t=1) with explicit Euler steps of +1/K on the same velocity field."""
    x = actions.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        v = velocity(model, prefix_pad_masks, past_key_values, x, i * dt)
        x = x + dt * v
    return x


@torch.no_grad()
def forward_cached(model, prefix_pad_masks, past_key_values, noise, num_steps: int):
    """Noise (t=1) -> actions (t=0): the euler_integrate loop on a precomputed prefix cache."""
    x = noise.clone()
    dt = -1.0 / num_steps
    for step in range(num_steps):
        v = velocity(model, prefix_pad_masks, past_key_values, x, 1.0 + step * dt)
        x = x + dt * v
    return x


@torch.no_grad()
def forward(model, images, img_masks, tokens, masks, noise, num_steps: int):
    return model.sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=num_steps)


# ------------------------------------------------------------------------- metrics
def rel(a: torch.Tensor, b: torch.Tensor, dims: slice) -> float:
    """||a - b|| / ||b|| over the given action dims."""
    num = torch.linalg.vector_norm((a - b)[..., dims].float())
    den = torch.linalg.vector_norm(b[..., dims].float()).clamp_min(1e-12)
    return float(num / den)


def max_abs(a: torch.Tensor, b: torch.Tensor, dims: slice) -> float:
    return float((a - b)[..., dims].float().abs().max())


def std(x: torch.Tensor, dims: slice) -> float:
    return float(x[..., dims].float().std())


def timeit(fn: Callable[[], object], n_iter: int, n_warmup: int = 2) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / n_iter


def repeat_inputs(images, img_masks, tokens, masks, batch_size: int):
    images_b = [im.expand(batch_size, *im.shape[1:]).contiguous() for im in images]
    img_masks_b = [m.expand(batch_size).contiguous() for m in img_masks]
    tokens_b = tokens.expand(batch_size, *tokens.shape[1:]).contiguous()
    masks_b = masks.expand(batch_size, *masks.shape[1:]).contiguous()
    return images_b, img_masks_b, tokens_b, masks_b


def mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.4f}±{arr.std():.4f}"


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3, help="number of noise seeds z to average over")
    parser.add_argument("--forward-steps", type=int, default=10)
    parser.add_argument("--reverse-steps", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--shift", type=float, default=0.3, help="normalised dx shift for the reference test")
    parser.add_argument("--shift-steps", type=int, default=10, help="chunk steps that receive the shift")
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument("--timing-batch", type=int, default=16)
    parser.add_argument("--compile", action="store_true", help="keep the checkpoint's torch.compile setting")
    parser.add_argument("--out", default=None, help="JSON output path")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    env_cfg = LiberoEnvConfig(task=args.task_suite, task_ids=[args.task_id])

    t0 = time.time()
    policy, preprocessor, postprocessor, env_pre, _env_post = load_policy(
        args.ckpt, args.dtype, n_action_steps=10, compile_model=args.compile, env_cfg=env_cfg
    )
    model = policy.model
    load_s = time.time() - t0
    mem_after_load = torch.cuda.memory_allocated() / 2**30
    print(f"policy loaded in {load_s:.1f}s, dtype={args.dtype}, allocated {mem_after_load:.2f} GiB")

    t0 = time.time()
    vec_env, batch, task_text = make_batch(env_cfg, preprocessor, env_pre, args.env_seed)
    print(f"env + batch in {time.time() - t0:.1f}s; task: {task_text!r}")
    print("batch:", {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in batch.items()})
    policy.reset()

    images, img_masks, tokens, masks = model_inputs(policy, batch)
    prefix_pad_masks, past_key_values = prefix_cache(model, images, img_masks, tokens, masks)
    print("prefix tokens:", int(prefix_pad_masks.sum()), "of", prefix_pad_masks.shape[1])

    K_fwd = args.forward_steps
    results: dict = {
        "dtype": args.dtype,
        "task": task_text,
        "forward_steps": K_fwd,
        "load_s": load_s,
        "rows": [],
        "shift_rows": [],
    }

    # --- sanity: determinism floor and cached-prefix equivalence -------------------
    gen = torch.Generator(device=device).manual_seed(0)
    z0 = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, generator=gen, device=device)
    a0 = forward(model, images, img_masks, tokens, masks, z0, K_fwd)
    a0_again = forward(model, images, img_masks, tokens, masks, z0, K_fwd)
    a0_cached = forward_cached(model, prefix_pad_masks, past_key_values, z0, K_fwd)
    results["determinism_rel_real"] = rel(a0_again, a0, REAL)
    results["cached_vs_sample_actions_rel_real"] = rel(a0_cached, a0, REAL)
    results["action_std_real"] = std(a0, REAL)
    results["action_std_pad"] = std(a0, PAD)
    print(
        f"determinism floor rel(real)={results['determinism_rel_real']:.2e}; "
        f"cached-prefix vs sample_actions rel(real)={results['cached_vs_sample_actions_rel_real']:.2e}; "
        f"normalised action std real={results['action_std_real']:.3f} pad={results['action_std_pad']:.3f}"
    )

    # --- (a)-(d) round trips over several noise seeds ------------------------------
    per_k: dict[int, dict[str, list[float]]] = {
        K: {
            k: []
            for k in [
                "z_real", "z_pad", "a_real", "a_pad", "a_real_maxabs", "zhat_std_real", "zhat_std_pad",
                "a_raw_maxabs", "a_real_padresample",
            ]
        }
        for K in args.reverse_steps
    }
    independent_noise_rel: list[float] = []  # how far apart actions from two unrelated noises are
    per_shift: dict[int, dict[str, list[float]]] = {
        K: {k: [] for k in ["aa_vs_ref_real", "aa_vs_a_real", "shift_kept", "zr_vs_z_real", "zr_std_real"]}
        for K in args.reverse_steps[:2]
    }
    for seed in range(args.n_seeds):
        gen = torch.Generator(device=device).manual_seed(1000 + seed)
        z = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, generator=gen, device=device)
        a = forward(model, images, img_masks, tokens, masks, z, K_fwd)
        z_other = torch.randn(z.shape, generator=gen, device=device)
        a_other = forward(model, images, img_masks, tokens, masks, z_other, K_fwd)
        independent_noise_rel.append(rel(a_other, a, REAL))
        for K in args.reverse_steps:
            z_hat = reverse(model, prefix_pad_masks, past_key_values, a, K)
            a2 = forward(model, images, img_masks, tokens, masks, z_hat, K_fwd)
            # DSBC-style: keep the reversed real dims, replace padding dims by fresh N(0, I)
            z_mix = z_hat.clone()
            z_mix[..., PAD] = torch.randn(z_mix[..., PAD].shape, generator=gen, device=device)
            a3 = forward(model, images, img_masks, tokens, masks, z_mix, K_fwd)
            r = per_k[K]
            r["a_real_padresample"].append(rel(a3, a, REAL))
            r["z_real"].append(rel(z_hat, z, REAL))
            r["z_pad"].append(rel(z_hat, z, PAD))
            r["a_real"].append(rel(a2, a, REAL))
            r["a_pad"].append(rel(a2, a, PAD))
            r["a_real_maxabs"].append(max_abs(a2, a, REAL))
            r["zhat_std_real"].append(std(z_hat, REAL))
            r["zhat_std_pad"].append(std(z_hat, PAD))
            try:  # un-normalised max abs error in raw LIBERO action units
                raw_a = postprocessor(a[..., REAL].clone())
                raw_a2 = postprocessor(a2[..., REAL].clone())
                r["a_raw_maxabs"].append(float((raw_a2 - raw_a).abs().max()))
            except Exception as exc:  # noqa: BLE001
                if seed == 0 and K == args.reverse_steps[0]:
                    print("postprocessor on chunk failed:", repr(exc))
                r["a_raw_maxabs"].append(float("nan"))
        # (e) shifted reference
        a_ref = a.clone()
        a_ref[:, : args.shift_steps, 0] += args.shift
        for K in per_shift:
            z_ref = reverse(model, prefix_pad_masks, past_key_values, a_ref, K)
            a_rr = forward(model, images, img_masks, tokens, masks, z_ref, K_fwd)
            s = per_shift[K]
            s["aa_vs_ref_real"].append(rel(a_rr, a_ref, REAL))
            s["aa_vs_a_real"].append(rel(a_rr, a, REAL))
            s["shift_kept"].append(float((a_rr - a)[:, : args.shift_steps, 0].mean() / args.shift))
            s["zr_vs_z_real"].append(rel(z_ref, z, REAL))
            s["zr_std_real"].append(std(z_ref, REAL))

    results["independent_noise_rel_a_real"] = independent_noise_rel
    print(f"\n=== round trip, dtype={args.dtype}, forward {K_fwd} steps, {args.n_seeds} noise seeds (mean±std)")
    print(f"context: rel(real) between actions from two independent noises = {mean_std(independent_noise_rel)}")
    print(f"{'K_rev':>5} | {'rel z real':>15} | {'rel z pad':>15} | {'rel a real':>15} | {'rel a pad':>15} | {'maxabs a real':>15} | {'maxabs raw':>13} | {'std zhat real':>15} | {'std zhat pad':>15} | {'rel a real,pad~N':>17}")
    for K, r in per_k.items():
        print(
            f"{K:>5} | {mean_std(r['z_real']):>15} | {mean_std(r['z_pad']):>15} | {mean_std(r['a_real']):>15} | "
            f"{mean_std(r['a_pad']):>15} | {mean_std(r['a_real_maxabs']):>15} | {mean_std(r['a_raw_maxabs']):>13} | "
            f"{mean_std(r['zhat_std_real']):>15} | {mean_std(r['zhat_std_pad']):>15} | {mean_std(r['a_real_padresample']):>17}"
        )
        results["rows"].append({"K_rev": K, **{k: v for k, v in r.items()}})

    print(f"\n=== shifted reference: +{args.shift} on normalised dx for the first {args.shift_steps} steps")
    print(f"{'K_rev':>5} | {'rel a2 vs ref':>15} | {'rel a2 vs a':>15} | {'shift kept':>15} | {'rel zref vs z':>15} | {'std zref real':>15}")
    for K, s in per_shift.items():
        print(
            f"{K:>5} | {mean_std(s['aa_vs_ref_real']):>15} | {mean_std(s['aa_vs_a_real']):>15} | "
            f"{mean_std(s['shift_kept']):>15} | {mean_std(s['zr_vs_z_real']):>15} | {mean_std(s['zr_std_real']):>15}"
        )
        results["shift_rows"].append({"K_rev": K, **{k: v for k, v in s.items()}})

    # --- timings -------------------------------------------------------------------
    print("\n=== timings")
    z1 = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, device=device)
    t_fwd_b1 = timeit(lambda: forward(model, images, img_masks, tokens, masks, z1, K_fwd), args.timing_iters)
    t_rev_b1 = timeit(lambda: reverse(model, prefix_pad_masks, past_key_values, a0, K_fwd), args.timing_iters)
    t_prefix_b1 = timeit(lambda: prefix_cache(model, images, img_masks, tokens, masks), args.timing_iters)
    print(f"batch 1 : sample_actions({K_fwd} steps) {t_fwd_b1:.1f} ms | reversal({K_fwd} steps, cached prefix) {t_rev_b1:.1f} ms | prefix+KV cache {t_prefix_b1:.1f} ms")
    results["timing_ms"] = {"b1_sample_actions": t_fwd_b1, "b1_reverse": t_rev_b1, "b1_prefix": t_prefix_b1}

    B = args.timing_batch
    images_b, img_masks_b, tokens_b, masks_b = repeat_inputs(images, img_masks, tokens, masks, B)
    zb = torch.randn(B, model.config.chunk_size, model.config.max_action_dim, device=device)
    t_fwd_bB = timeit(lambda: forward(model, images_b, img_masks_b, tokens_b, masks_b, zb, K_fwd), args.timing_iters)
    ppm_b, pkv_b = prefix_cache(model, images_b, img_masks_b, tokens_b, masks_b)
    ab = forward(model, images_b, img_masks_b, tokens_b, masks_b, zb, K_fwd)
    t_rev_bB = timeit(lambda: reverse(model, ppm_b, pkv_b, ab, K_fwd), args.timing_iters)
    print(f"batch {B}: sample_actions({K_fwd} steps) {t_fwd_bB:.1f} ms | reversal({K_fwd} steps, cached prefix) {t_rev_bB:.1f} ms")
    results["timing_ms"].update({f"b{B}_sample_actions": t_fwd_bB, f"b{B}_reverse": t_rev_bB})

    results["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "after_load_gib": mem_after_load,
    }
    print(f"GPU {results['gpu']['name']}: max allocated {results['gpu']['max_allocated_gib']:.2f} GiB, max reserved {results['gpu']['max_reserved_gib']:.2f} GiB")

    vec_env.close()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
