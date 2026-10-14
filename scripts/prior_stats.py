"""Summarize what a trained LeaP prior predicts over a dataset.

Writes JSON with per-action-dimension statistics of mu and sigma (normalized
action units), the Gaussian NLL of the true chunk, and how far mu sits from the
true next action versus the current state. The NoPrior baseline yields mu = 0,
sigma = 1 by construction.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from leap.robotwin.checkpoint import load_policy
from leap.robotwin.data import RobotDataset


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    policy = load_policy(args.checkpoint, device=args.device)
    dataset = RobotDataset(args.data, horizon=policy.horizon, n_obs_steps=policy.n_obs_steps,
                           n_action_steps=policy.n_action_steps, val_ratio=0.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    mus, sigmas, nlls, d_action, d_state, d_action_chunk = [], [], [], [], [], []
    for batch in loader:
        obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
        nobs = policy.normalizer.normalize(obs)
        nact = policy.normalizer["action"].normalize(batch["action"].to(args.device))
        mu, log_var, _ = policy._encode_obs(nobs)
        sigma = torch.exp(0.5 * log_var)
        # Current joint state expressed in the *action* normalizer's coordinates, like mu.
        current = policy.normalizer["action"].normalize(obs["agent_pos"][:, policy.n_obs_steps - 1])
        assert current.shape[-1] == mu.shape[-1], "state and action dimensions differ"
        first_action = nact[:, policy.n_obs_steps - 1]              # first executed action
        nll = 0.5 * (log_var[:, None] + (nact - mu[:, None]) ** 2 / sigma[:, None] ** 2 + math.log(2 * math.pi))
        mus.append(mu.cpu()); sigmas.append(sigma.cpu()); nlls.append(nll.mean(dim=(1, 2)).cpu())
        d_action.append((mu - first_action).abs().cpu())
        d_state.append((mu - current).abs().cpu())
        d_action_chunk.append((mu[:, None] - nact).abs().mean(1).cpu())
    mu = torch.cat(mus); sigma = torch.cat(sigmas)
    summary = {
        "checkpoint": str(args.checkpoint), "data": args.data, "windows": len(dataset),
        "prior_type": policy.prior_type,
        "mu_mean_per_dim": mu.mean(0).tolist(), "mu_std_per_dim": mu.std(0).tolist(),
        "sigma_mean_per_dim": sigma.mean(0).tolist(), "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()), "sigma_mean": float(sigma.mean()),
        "nll_per_window_mean": float(torch.cat(nlls).mean()),
        "abs_mu_minus_first_action_mean": float(torch.cat(d_action).mean()),
        "abs_mu_minus_current_state_mean": float(torch.cat(d_state).mean()),
        "abs_mu_minus_chunk_mean": float(torch.cat(d_action_chunk).mean()),
        "note": "normalized action units in [-1, 1]; NoPrior has mu=0, sigma=1 by construction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("per_dim")}, indent=2))


if __name__ == "__main__":
    main()
