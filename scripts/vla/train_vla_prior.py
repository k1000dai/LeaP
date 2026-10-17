"""Train the VLA prior (forward model of pi0.5's flow source) on an offline reversal dataset.

Reads every ``task_*.npz`` written by scripts/vla/build_reversal_dataset.py (feat [N, 4104],
noise [N, 50, 7], chunk_pad_mask [N, 50], demo_id [N], instr_<modality> [D, V, 768], ...),
holds out --val-fraction of the demos of every task, standardises features and instruction
embeddings with the statistics of the training split, and fits ``leap.vla.prior.VLAPrior``
with Adam on the masked Gaussian NLL of the reversal noises.  With --instr-mode concat the
steering modality of every training sample is drawn uniformly from --modalities and its
variant uniformly from the variants stored for that demo; validation uses variant 0 of each
modality (the canonical instruction that evaluation builds) and reports one NLL per modality.

Outputs under --out: metrics.jsonl (train/val NLL every --log-every steps), run.json (args,
split, statistics and the reference NLLs of the N(0, I) source and of an empirical
per-element Gaussian), prior.pt (final weights) and best.pt (lowest validation NLL).

Example:
    python scripts/vla/train_vla_prior.py --data data/vla/libero_spatial_reversal \
        --out outputs/vla/prior-concat --instr-mode concat --modalities text,sketch,audio --device cuda
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leap.vla.prior import STD_FLOOR, VLAPrior, gaussian_nll, standard_normal_nll  # noqa: E402

MODALITIES = ("text", "sketch", "audio")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Directory holding task_*.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instr-mode", choices=("none", "concat"), default="concat")
    parser.add_argument("--modalities", default="text,sketch,audio",
                        help="Comma-separated subset of text,sketch,audio (ignored with --instr-mode none)")
    parser.add_argument("--steps", type=int, default=12500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", default="1024,1024", help="Comma-separated MLP widths")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=0.0, help="Weight of an extra masked MSE term")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of demos held out per task")
    parser.add_argument("--val-batch", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Retrain even if <out>/prior.pt exists")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch < 1 or args.log_every < 1 or args.val_batch < 1:
        parser.error("steps, batch, log-every and val-batch must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("val-fraction must be in [0, 1)")
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if args.instr_mode == "concat":
        unknown = [m for m in modalities if m not in MODALITIES]
        if unknown or not modalities:
            parser.error(f"--modalities must be a non-empty subset of {MODALITIES}, got {modalities}")
    else:
        modalities = []
    args.modalities = modalities
    args.hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    return args


# ---- data ----------------------------------------------------------------------------------------------
def task_number(path):
    return int(path.stem.split("_")[-1])


def load_tasks(data_dir, modalities):
    """One dict per task_*.npz with the arrays used here; shapes are validated against each other."""
    paths = sorted(Path(data_dir).glob("task_*.npz"), key=task_number)
    if not paths:
        raise FileNotFoundError(f"no task_*.npz under {data_dir}")
    tasks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            task = {"path": str(path), "task_id": int(npz["task_id"]), "task_language": str(npz["task_language"]),
                    "feat": np.asarray(npz["feat"], dtype=np.float32),
                    "noise": np.asarray(npz["noise"], dtype=np.float32),
                    "pad": np.asarray(npz["chunk_pad_mask"], dtype=bool),
                    "demo_id": np.asarray(npz["demo_id"], dtype=np.int64),
                    "recon_err": np.asarray(npz["recon_err"], dtype=np.float32) if "recon_err" in npz else None}
            for modality in modalities:
                key = f"instr_{modality}"
                if key not in npz:
                    raise KeyError(f"{path} has no {key}")
                task[key] = np.asarray(npz[key], dtype=np.float32)
        n = task["feat"].shape[0]
        if task["feat"].ndim != 2 or task["noise"].shape[:1] != (n,) or task["noise"].ndim != 3:
            raise ValueError(f"{path}: feat {task['feat'].shape} / noise {task['noise'].shape} mismatch")
        if task["pad"].shape != task["noise"].shape[:2] or task["demo_id"].shape != (n,):
            raise ValueError(f"{path}: chunk_pad_mask {task['pad'].shape} / demo_id {task['demo_id'].shape} mismatch")
        for modality in modalities:
            instr = task[f"instr_{modality}"]
            if instr.ndim != 3 or instr.shape[0] <= task["demo_id"].max():
                raise ValueError(f"{path}: instr_{modality} {instr.shape} does not index every demo_id")
        tasks.append(task)
    return tasks


def split_demos(tasks, val_fraction, seed):
    """Per task, hold out round(val_fraction * D) demos (at least one when D > 1); returns is_val per task."""
    rng = np.random.default_rng(seed)
    masks, split = [], {}
    for task in tasks:
        demos = np.unique(task["demo_id"])
        n_val = int(round(val_fraction * len(demos)))
        if val_fraction > 0 and len(demos) > 1:
            n_val = min(max(1, n_val), len(demos) - 1)
        val_demos = rng.permutation(demos)[:n_val]
        masks.append(np.isin(task["demo_id"], val_demos))
        split[str(task["task_id"])] = {"demos": int(len(demos)), "val_demos": sorted(int(d) for d in val_demos),
                                       "samples": int(len(task["demo_id"])), "val_samples": int(masks[-1].sum())}
    return masks, split


def stack_dataset(tasks, val_masks, modalities):
    """Concatenate the tasks; instruction tables are stacked over demos with per-row variant counts."""
    data = {"feat": np.concatenate([t["feat"] for t in tasks]),
            "noise": np.concatenate([t["noise"] for t in tasks]),
            "pad": np.concatenate([t["pad"] for t in tasks]),
            "is_val": np.concatenate(val_masks),
            "task_index": np.concatenate([np.full(len(t["demo_id"]), i) for i, t in enumerate(tasks)])}
    offsets = np.cumsum([0] + [int(t["demo_id"].max()) + 1 for t in tasks])
    data["demo_row"] = np.concatenate([t["demo_id"] + offsets[i] for i, t in enumerate(tasks)])
    data["instr"], data["instr_count"] = {}, {}
    for modality in modalities:
        tables = [t[f"instr_{modality}"] for t in tasks]
        v_max = max(table.shape[1] for table in tables)
        stacked = np.zeros((offsets[-1], v_max, tables[0].shape[2]), dtype=np.float32)
        counts = np.zeros(offsets[-1], dtype=np.int64)
        for i, table in enumerate(tables):
            rows = slice(offsets[i], offsets[i] + table.shape[0])
            stacked[rows, :table.shape[1]] = table
            counts[rows] = table.shape[1]
        data["instr"][modality] = stacked
        data["instr_count"][modality] = counts
    return data


def compute_stats(data, modalities):
    """Feature mean/std over training samples; instruction mean/std pooled over modalities and variants of training demos."""
    train = ~data["is_val"]
    feat = data["feat"][train].astype(np.float64)
    stats = {"feat_mean": feat.mean(axis=0).astype(np.float32),
             "feat_std": np.maximum(feat.std(axis=0), STD_FLOOR).astype(np.float32)}
    if modalities:
        rows = np.unique(data["demo_row"][train])
        pooled = []
        for modality in modalities:
            table, counts = data["instr"][modality], data["instr_count"][modality]
            pooled += [table[r, :counts[r]] for r in rows]
        pooled = np.concatenate(pooled).astype(np.float64)
        stats["instr_mean"] = pooled.mean(axis=0).astype(np.float32)
        stats["instr_std"] = np.maximum(pooled.std(axis=0), STD_FLOOR).astype(np.float32)
    return stats


def gather_instr(data, idx, modalities, rng=None, modality=None):
    """[len(idx), 768] embeddings: modality drawn uniformly (or fixed) and the variant uniformly (or 0 without rng)."""
    rows = data["demo_row"][idx]
    chosen = (np.full(len(idx), modalities.index(modality)) if modality is not None
              else rng.integers(len(modalities), size=len(idx)))
    out = np.zeros((len(idx), next(iter(data["instr"].values())).shape[2]), dtype=np.float32)
    for k, name in enumerate(modalities):
        pick = np.flatnonzero(chosen == k)
        if not len(pick):
            continue
        counts = data["instr_count"][name][rows[pick]]
        variant = (rng.integers(counts) if rng is not None else np.zeros(len(pick), dtype=np.int64))
        out[pick] = data["instr"][name][rows[pick], variant]
    return out


# ---- evaluation ------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, tensors, idx, modalities, val_batch):
    """Validation NLL per modality (variant 0) and their mean; {"val_nll": nan} when idx is empty."""
    if len(idx) == 0:
        return {"val_nll": float("nan")}
    model.eval()
    results = {}
    for modality in (modalities or [None]):
        total, count = 0.0, 0
        for start in range(0, len(idx), val_batch):
            chunk = idx[start:start + val_batch]
            feat, noise, pad = (tensors[k][chunk] for k in ("feat", "noise", "pad"))
            instr = None
            if modality is not None:
                instr = torch.as_tensor(gather_instr(data, chunk, modalities, modality=modality), device=feat.device)
            mu, log_var = model(feat, instr)
            weight = int((~pad).sum())
            total += float(gaussian_nll(mu, log_var, noise, pad)) * weight
            count += weight
        results[f"val_nll_{modality}" if modality else "val_nll"] = total / max(count, 1)
    if modalities:
        results["val_nll"] = float(np.mean([results[f"val_nll_{m}"] for m in modalities]))
    return results


@torch.no_grad()
def reference_nlls(tensors, train_idx, val_idx):
    """NLL of the validation labels under N(0, I) and under the per-element Gaussian fitted on the training labels."""
    if len(val_idx) == 0:
        return {}
    noise, pad = tensors["noise"][val_idx], tensors["pad"][val_idx]
    keep = (~tensors["pad"][train_idx]).to(torch.float32).unsqueeze(-1)
    train_noise = tensors["noise"][train_idx]
    count = keep.sum(dim=0).clamp_min(1.0)
    mean = (train_noise * keep).sum(dim=0) / count
    var = (((train_noise - mean) ** 2) * keep).sum(dim=0) / count
    mu = mean.unsqueeze(0).expand_as(noise)
    log_var = torch.log(var.clamp_min(STD_FLOOR ** 2)).unsqueeze(0).expand_as(noise)
    return {"gauss_val_nll": float(standard_normal_nll(noise, pad)),
            "empirical_val_nll": float(gaussian_nll(mu, log_var, noise, pad)),
            "label_mean_abs": float(mean.abs().mean()), "label_std_mean": float(var.sqrt().mean())}


def recon_summary(tasks):
    values = [t["recon_err"][np.isfinite(t["recon_err"])] for t in tasks if t["recon_err"] is not None]
    values = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    if not len(values):
        return {"measured": 0}
    return {"measured": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


# ---- training ------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prior.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out / 'prior.pt'} exists; pass --overwrite to retrain")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    tasks = load_tasks(args.data, args.modalities)
    val_masks, split = split_demos(tasks, args.val_fraction, args.seed)
    data = stack_dataset(tasks, val_masks, args.modalities)
    stats = compute_stats(data, args.modalities)
    train_idx = np.flatnonzero(~data["is_val"])
    val_idx = np.flatnonzero(data["is_val"])
    horizon, action_dim = data["noise"].shape[1:]
    tensors = {"feat": torch.as_tensor(data["feat"], device=device),
               "noise": torch.as_tensor(data["noise"], device=device),
               "pad": torch.as_tensor(data["pad"], device=device)}
    print(f"loaded {len(tasks)} tasks: {len(train_idx)} train / {len(val_idx)} val samples, "
          f"feat {data['feat'].shape[1]}, chunk {horizon}x{action_dim}, modalities {args.modalities or 'none'}",
          flush=True)

    model = VLAPrior(feat_dim=data["feat"].shape[1], instr_dim=next(iter(data["instr"].values())).shape[2]
                     if args.modalities else 768, hidden=args.hidden, horizon=horizon, action_dim=action_dim,
                     instr_mode=args.instr_mode, dropout=args.dropout)
    model.set_stats(stats["feat_mean"], stats["feat_std"], stats.get("instr_mean"), stats.get("instr_std"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    references = reference_nlls(tensors, train_idx, val_idx)
    run = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "tasks": [{"task_id": t["task_id"], "task_language": t["task_language"], "path": t["path"]} for t in tasks],
           "split": split, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
           "model": model.config(), "parameters": sum(p.numel() for p in model.parameters()),
           "references": references, "recon_err": recon_summary(tasks),
           "feat_std_range": [float(stats["feat_std"].min()), float(stats["feat_std"].max())]}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    for key, value in references.items():
        print(f"reference {key}: {value:.4f}", flush=True)

    extra = {"args": run["args"], "split": split, "references": references, "modalities": args.modalities}
    best = {"step": 0, "val_nll": math.inf}
    window, step, order, cursor, started = [], 0, rng.permutation(train_idx), 0, time.time()
    with (out / "metrics.jsonl").open("w") as log:
        while step < args.steps:
            if cursor + args.batch > len(order):
                order, cursor = rng.permutation(train_idx), 0
            idx = order[cursor:cursor + args.batch]
            cursor += args.batch
            model.train()
            feat, noise, pad = (tensors[k][idx] for k in ("feat", "noise", "pad"))
            instr = (torch.as_tensor(gather_instr(data, idx, args.modalities, rng=rng), device=device)
                     if args.modalities else None)
            mu, log_var = model(feat, instr)
            nll = gaussian_nll(mu, log_var, noise, pad)
            loss = model.nll(mu, log_var, noise, pad, args.mse_weight) if args.mse_weight else nll
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            window.append((float(nll.detach()), float(loss.detach())))
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {"step": step, "train_nll": float(np.mean([w[0] for w in window])),
                       "train_loss": float(np.mean([w[1] for w in window])), "elapsed": time.time() - started}
                row.update(evaluate(model, data, tensors, val_idx, args.modalities, args.val_batch))
                window = []
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if len(val_idx) and row["val_nll"] < best["val_nll"]:
                    best = {"step": step, "val_nll": row["val_nll"]}
                    model.save(out / "best.pt", dict(extra, step=step, val_nll=row["val_nll"]))
    final = evaluate(model, data, tensors, val_idx, args.modalities, args.val_batch)
    model.save(out / "prior.pt", dict(extra, step=step, val_nll=final["val_nll"]))
    if not len(val_idx) or not (out / "best.pt").exists():
        model.save(out / "best.pt", dict(extra, step=step, val_nll=final["val_nll"]))
        best = {"step": step, "val_nll": final["val_nll"]}
    run.update({"final": final, "best": best, "steps": step, "seconds": time.time() - started})
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(f"final val NLL {final['val_nll']:.4f} (best {best['val_nll']:.4f} at step {best['step']}); "
          f"references {json.dumps(references)}", flush=True)
    print(f"Saved {out / 'prior.pt'} and {out / 'best.pt'} ({step} updates)", flush=True)


if __name__ == "__main__":
    main()
