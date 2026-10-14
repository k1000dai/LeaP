"""Offline steerability probe: how much of an inference-time mu shift reaches the action?

For dataset observations, predict actions with mu shifted by delta along one
action dimension and report the mean change of the first executed action in
that dimension (pass-through gain = change / delta), the change leaked into
other dimensions, and the across-sample spread under sigma scaling. Everything
is in normalized action units ([-1, 1]). No simulator involved.
"""
import argparse
import json
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
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 3, 7, 10])
    parser.add_argument("--deltas", type=float, nargs="+", default=[-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--sigma-scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    policy = load_policy(args.checkpoint, device=args.device)
    dataset = RobotDataset(args.data, horizon=policy.horizon, n_obs_steps=policy.n_obs_steps,
                           n_action_steps=policy.n_action_steps, val_ratio=0.0)
    rng = np.random.default_rng(args.seed)
    subset = torch.utils.data.Subset(dataset, rng.choice(len(dataset), size=min(args.windows, len(dataset)), replace=False))
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    A, first = policy.action_dim, policy.n_obs_steps - 1

    def predict(intervention):
        policy.set_prior_intervention(intervention)
        outs = []
        for batch in loader:
            obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
            torch.manual_seed(args.seed)
            pred = policy.predict_action(obs)["action_pred"]
            outs.append(policy.normalizer["action"].normalize(pred)[:, first].cpu())
        policy.set_prior_intervention(None)
        return torch.cat(outs)

    base = predict({"sigma_scale": 0})           # deterministic reference (source = mu)
    result = {"checkpoint": str(args.checkpoint), "windows": len(subset), "action_dim": A, "mu_shift": {}, "sigma_scale": {}}
    for d in args.dims:
        rows = []
        for delta in args.deltas:
            shift = [0.0] * A; shift[d] = delta
            out = predict({"sigma_scale": 0, "mu_shift": shift})
            diff = out - base
            rows.append({"delta": delta, "gain_dim": float(diff[:, d].mean() / delta),
                         "gain_dim_std": float(diff[:, d].std() / abs(delta)),
                         "leak_other_dims_mean_abs": float(diff[:, [i for i in range(A) if i != d]].abs().mean())})
        result["mu_shift"][str(d)] = rows
    for scale in args.sigma_scales:
        samples = []
        for s in range(4):
            policy.set_prior_intervention({"sigma_scale": scale})
            outs = []
            for batch in loader:
                obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
                torch.manual_seed(1000 + s)
                outs.append(policy.normalizer["action"].normalize(policy.predict_action(obs)["action_pred"])[:, first].cpu())
            samples.append(torch.cat(outs))
        policy.set_prior_intervention(None)
        stack = torch.stack(samples)
        result["sigma_scale"][str(scale)] = {"action_spread_across_samples": float(stack.std(0).mean()),
                                            "mean_abs_shift_from_deterministic": float((stack.mean(0) - base).abs().mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for d, rows in result["mu_shift"].items():
        print(f"dim {d}: " + "  ".join(f"δ={r['delta']:+.2f} gain={r['gain_dim']:.2f} leak={r['leak_other_dims_mean_abs']:.3f}" for r in rows))
    for s, r in result["sigma_scale"].items():
        print(f"sigma×{s}: spread={r['action_spread_across_samples']:.4f} mean|Δ| vs deterministic={r['mean_abs_shift_from_deterministic']:.4f}")


if __name__ == "__main__":
    main()
