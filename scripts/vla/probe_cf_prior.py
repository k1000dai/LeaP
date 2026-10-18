#!/usr/bin/env python
"""Does a counterfactual prior (instr_mode concat) encode which bowl the instruction names?

For every LIBERO-Spatial task that has both a bowl-1 and a bowl-2 file in the counterfactual reversal dataset,
the held-out rows (the trainer's split, seed 0) of each bowl are scored with the row's own instruction and with
the other bowl's instruction for the same observation features: Gaussian NLL of the reversal label (mean over
the 50 x 7 elements), the fraction of rows whose own instruction scores better, and the mean |mu_own - mu_swapped|
in units of the predicted sigma.  Text swaps the other file's prompt; sketch swaps the other bowl's sketch of the
same init state (only where that episode succeeded for both bowls).

Usage (VLA venv, CPU is enough):
    python scripts/vla/probe_cf_prior.py --prior outputs/vla_cf/prior-cf-concat/best.pt \
        --data data/vla/libero_spatial_counterfactual
"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leap.vla.prior import VLAPrior  # noqa: E402


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_vla_prior", ROOT / "scripts/vla/train_vla_prior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_nll(mu, log_var, noise):
    return 0.5 * (log_var + (noise - mu) ** 2 / torch.exp(log_var) + math.log(2 * math.pi))


@torch.no_grad()
def nll(prior, feat, instr, noise, batch=1024):
    out = []
    for i in range(0, len(feat), batch):
        mu, log_var = prior(torch.as_tensor(feat[i:i + batch]), torch.as_tensor(instr[i:i + batch]))
        out.append(row_nll(mu, log_var, torch.as_tensor(noise[i:i + batch])).mean(dim=(1, 2)))
    return torch.cat(out).numpy()


@torch.no_grad()
def mu_sigma(prior, feat, instr):
    mu, log_var = prior(torch.as_tensor(feat), torch.as_tensor(instr))
    return mu.numpy(), torch.exp(0.5 * log_var).numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modalities", default="text,sketch")
    args = parser.parse_args(argv)
    trainer = load_trainer()
    prior, _ = VLAPrior.load(args.prior)
    if prior.instr_mode != "concat":
        raise SystemExit(f"{args.prior}: instr_mode {prior.instr_mode!r} takes no instruction")
    tasks = trainer.load_tasks(args.data, ["text", "sketch", "audio"])
    is_val = trainer.split_demos(tasks, args.val_fraction, args.seed)[0]
    by_file = {int(Path(t["path"]).stem.split("_")[1]): (t, v) for t, v in zip(tasks, is_val)}
    modalities = args.modalities.split(",")
    head = " | ".join(f"{m}: NLL own / swapped (Δ) | own<swapped % | abs(Δμ)/σ" for m in modalities)
    print(f"| task | bowl | held-out rows | {head} |")
    print("|---|---|---|" + "---|---|---|" * len(modalities))
    for t in range(10):
        if 100 + t not in by_file or 200 + t not in by_file:
            continue
        files = {1: by_file[100 + t], 2: by_file[200 + t]}
        for bowl in (1, 2):
            task, val = files[bowl]
            other = files[3 - bowl][0]
            rows = np.flatnonzero(val)
            if not len(rows):
                continue
            cells = []
            for m in modalities:
                feat, noise = task["feat"][rows], task["noise"][rows]
                own = task[f"instr_{m}"][task["demo_row"][rows], 0]
                if m == "text":
                    swapped = np.repeat(other[f"instr_{m}"][0:1, 0], len(rows), axis=0)
                else:
                    episode = task["demo_id"][rows] % 1000
                    other_episodes = np.unique(other["demo_id"]) % 1000
                    keep = np.isin(episode, other_episodes)
                    if not keep.any():
                        cells.append("n/a | | ")
                        continue
                    swapped = other[f"instr_{m}"][np.searchsorted(other_episodes, episode[keep]), 0]
                    feat, noise, own = feat[keep], noise[keep], own[keep]
                nll_own, nll_swapped = nll(prior, feat, own, noise), nll(prior, feat, swapped, noise)
                mu_own, sigma_own = mu_sigma(prior, feat, own)
                mu_swapped, _ = mu_sigma(prior, feat, swapped)
                cells.append(f"{nll_own.mean():.3f} / {nll_swapped.mean():.3f} ({nll_swapped.mean() - nll_own.mean():+.3f}) | "
                             f"{(nll_own < nll_swapped).mean() * 100:.0f} | {np.abs(mu_own - mu_swapped).mean() / sigma_own.mean():.2f}")
            print(f"| {t} | {bowl} | {len(rows)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
