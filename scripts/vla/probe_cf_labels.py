#!/usr/bin/env python
"""Do the counterfactual reversal labels carry the bowl, and how much of that does a prior reproduce?  (CPU, offline.)

Diagnoses whether the weak steering of the counterfactual prior is a LABEL problem (the reversed noises of the
bowl-1 and bowl-2 rollouts of the same init state hardly differ, or differ inconsistently) or a FIT problem (the
labels separate the bowls but the prior's mu does not move with the instruction).  Three parts, per task that has
both a bowl-1 (``task_10t.npz``) and a bowl-2 (``task_20t.npz``) file in the counterfactual reversal dataset:

(a) label geometry   matched pairs by init state (``demo_id % 100``) at the same query index q (0 = the first
    query, where both rollouts see the same observation): delta = noise(bowl 2) - noise(bowl 1).  Reported
    against the within-bowl spread sigma (per-element std across init states, averaged over the two bowls):
    mean |delta| / sigma, ||delta|| / ||sigma||, the effect size d' = RMS over elements of mean(delta) / sigma, the
    pairwise cosine of delta across init states (a shared bowl direction?), the share ||mean delta|| / mean ||delta||,
    which chunk steps and action dims carry it, and the cosine between the tasks' mean directions.
(b) probes           held-out-init-state classification of the bowl from the noise label (leave-one-state-out per task,
    20 folds of states pooled): a ridge classifier (lambda by inner 5-fold over states) and the mean-direction
    classifier (sign of the projection on centroid2 - centroid1), on q = 0 and on q <= --max-query, per task and
    pooled over the tasks.
(c) prior            for every --prior: delta_mu = mu(feat, e_bowl2) - mu(feat, e_bowl1) (``VLAPrior.steered`` with the
    --gains; text / sketch / audio embeddings of the row's task and demo) against delta: ||delta_mu|| / ||delta||, the
    projection (delta_mu . delta) / ||delta||^2 (the fraction of the label shift reproduced along its direction), the
    cosine, the mean-direction classifier applied to mu (offline bowl choice), the R^2 of mu on the labels (train and
    held-out rows), and for an instr_mode "none" prior the observation-only reference mu(feat_bowl2) - mu(feat_bowl1)
    (at q = 0 a leak check: the two rows show the same scene).  Also the cosine between the standardised bowl-1 and
    bowl-2 embeddings (how different the instruction inputs are).

Held-out rows follow the trainer's split (--val-fraction, --seed, --val-split as ``train_vla_prior.py``) unless
--val-init-states lists the held-out init states.  Writes a JSON (--out) and prints markdown tables.

Usage (remote, VLA venv, CPU):
    python scripts/vla/probe_cf_labels.py --data data/vla/libero_spatial_counterfactual \\
        --prior outputs/vla_cf/prior-cf-concat/best.pt outputs/vla_cf/prior-cf-none/best.pt \\
        --gains 1,2,3 --out outputs/vla_cf/probe_cf_labels.json
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leap.vla.prior import VLAPrior  # noqa: E402

MODALITIES = ("text", "sketch", "audio")
INIT_STATES_PER_REPEAT = 100     # counterfactual.init_state_of: demo_id % 100 (repeats live in the hundreds)
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
INNER_FOLDS = 5
OUTER_FOLDS = 20
MIN_STATES = 3
STEP_BANDS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_vla_prior", ROOT / "scripts/vla/train_vla_prior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- data ----------------------------------------------------------------------------------------------------------
def file_index(path):
    return int(Path(path).stem.split("_")[1])


def query_index(demo_id, t):
    """Rank of ``t`` within its demo: 0 for the first recorded query of every rollout."""
    demo_id, t = np.asarray(demo_id), np.asarray(t)
    q = np.zeros(len(t), dtype=np.int64)
    for demo in np.unique(demo_id):
        rows = np.flatnonzero(demo_id == demo)
        q[rows[np.argsort(t[rows], kind="stable")]] = np.arange(len(rows))
    return q


def load_cf_dataset(data_dir, modalities=MODALITIES):
    """``{index: file}`` for every ``task_{n}.npz`` (``n = 100 * bowl + task``); plain demo files (n < 100) are kept too.

    Each file holds feat, noise, demo_id, t, the instr_<modality> tables present, ``demo_row`` (row of the demo in
    the instruction tables), ``init_state`` (``demo_id % 100``) and ``query`` (rank of t within the demo).
    """
    files = {}
    paths = sorted(Path(data_dir).glob("task_*.npz"), key=file_index)
    if not paths:
        raise FileNotFoundError(f"no task_*.npz under {data_dir}")
    for path in paths:
        n = file_index(path)
        with np.load(path, allow_pickle=False) as npz:
            data = {"path": str(path), "index": n, "bowl": n // 100, "task": n % 100,
                    "task_id": int(npz["task_id"]) if "task_id" in npz else n % 100,
                    "feat": np.asarray(npz["feat"], dtype=np.float32), "noise": np.asarray(npz["noise"], dtype=np.float32),
                    "demo_id": np.asarray(npz["demo_id"], dtype=np.int64), "t": np.asarray(npz["t"], dtype=np.int64)}
            for modality in modalities:
                key = f"instr_{modality}"
                if key in npz and np.any(npz[key]):
                    data[key] = np.asarray(npz[key], dtype=np.float32)
        if data["feat"].shape[0] != data["noise"].shape[0] or data["demo_id"].shape[0] != data["t"].shape[0] != data["feat"].shape[0]:
            raise ValueError(f"{path}: feat / noise / demo_id / t row counts differ")
        unique = np.unique(data["demo_id"])
        data["demo_row"] = np.searchsorted(unique, data["demo_id"])
        data["init_state"] = data["demo_id"] % INIT_STATES_PER_REPEAT
        data["query"] = query_index(data["demo_id"], data["t"])
        files[n] = data
    return files


def val_masks(files, val_fraction, seed, val_init_states=None, val_split="demo"):
    """``{index: bool [N]}`` held-out rows: the trainer's split (same file order and rng) or an init-state list.

    ``val_split`` is passed to ``train_vla_prior.split_demos`` when that trainer accepts it ("episode" holds out init
    states shared by the bowl files of a task); an older trainer only knows the per-file demo split.
    """
    if val_init_states is not None:
        return {n: np.isin(f["init_state"], list(val_init_states)) for n, f in files.items()}
    trainer = load_trainer()
    order = sorted(files)
    light = [{"demo_id": files[n]["demo_id"], "task_id": files[n]["task_id"]} for n in order]
    if val_split != "demo":
        if "val_split" not in trainer.split_demos.__code__.co_varnames:
            raise SystemExit(f"this train_vla_prior.py has no val_split option (asked for {val_split!r})")
        masks, _ = trainer.split_demos(light, val_fraction, seed, val_split=val_split)
    else:
        masks, _ = trainer.split_demos(light, val_fraction, seed)
    return dict(zip(order, masks))


def paired_tasks(files):
    """Suite task ids with both a bowl-1 and a bowl-2 file."""
    return sorted(t for t in range(100) if 100 + t in files and 200 + t in files)


def matched_pairs(file1, file2, max_query):
    """``[(init_state, q, row1, row2)]``: rows of the two files with the same init state and query index q <= max_query."""
    def by_key(data):
        table = {}
        for row, (s, q) in enumerate(zip(data["init_state"], data["query"])):
            if q <= max_query:
                table.setdefault((int(s), int(q)), []).append(row)
        return table
    one, two = by_key(file1), by_key(file2)
    pairs = []
    for key in sorted(one):
        for r1 in one[key]:
            for r2 in two.get(key, []):
                pairs.append((key[0], key[1], r1, r2))
    return pairs


# ---- (a) label geometry ----------------------------------------------------------------------------------------------
def flat(x):
    return np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)


def pairwise_cosine(vectors):
    """Mean cosine over all pairs of rows (NaN with fewer than two rows)."""
    v = flat(vectors)
    if v.shape[0] < 2:
        return float("nan")
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    gram = v @ v.T
    return float(gram[np.triu_indices(v.shape[0], 1)].mean())


def within_bowl_sigma(noise1, noise2):
    """Per-element std across init states, averaged (in variance) over the two bowls: [H, A]."""
    var = 0.5 * (noise1.var(axis=0) + noise2.var(axis=0)) if len(noise1) > 1 and len(noise2) > 1 else \
        np.maximum(noise1.var(axis=0), noise2.var(axis=0))
    return np.sqrt(np.maximum(var, 1e-12))


def label_shift(file1, file2, pairs, queries=(0,)):
    """Label-shift statistics per query index: delta = noise2 - noise1 against the within-bowl spread."""
    out = {}
    for q in queries:
        chosen = [p for p in pairs if p[1] == q]
        if not chosen:
            continue
        rows1 = np.unique([p[2] for p in chosen])
        rows2 = np.unique([p[3] for p in chosen])
        n1, n2 = file1["noise"][rows1], file2["noise"][rows2]
        sigma = within_bowl_sigma(n1, n2)
        delta = np.stack([file2["noise"][r2] - file1["noise"][r1] for _, _, r1, r2 in chosen]).astype(np.float64)
        scaled = delta / sigma
        mean_delta = delta.mean(axis=0)
        out[q] = {
            "pairs": len(chosen), "init_states": len({p[0] for p in chosen}),
            "delta_norm": float(np.linalg.norm(flat(delta), axis=1).mean()),
            "sigma_norm": float(np.linalg.norm(sigma)),
            "label_rms": float(np.sqrt(np.mean(np.concatenate([n1, n2]) ** 2))),
            "within_pair_norm": float(np.mean([np.linalg.norm(n1[i] - n1[j]) for i in range(len(n1)) for j in range(i + 1, len(n1))]
                                              + [np.linalg.norm(n2[i] - n2[j]) for i in range(len(n2)) for j in range(i + 1, len(n2))]))
            if len(n1) > 1 or len(n2) > 1 else float("nan"),
            "mean_abs_delta_over_sigma": float(np.abs(scaled).mean()),
            "delta_over_sigma_norm": float(np.linalg.norm(flat(delta), axis=1).mean() / np.linalg.norm(sigma)),
            "d_prime": float(np.sqrt(np.mean((mean_delta / sigma) ** 2))),
            "pairwise_cos": pairwise_cosine(delta),
            "shared_fraction": float(np.linalg.norm(mean_delta) / np.linalg.norm(flat(delta), axis=1).mean()),
            "per_step": np.abs(scaled).mean(axis=(0, 2)).tolist(),
            "per_dim": np.abs(scaled).mean(axis=(0, 1)).tolist(),
            "mean_delta": mean_delta,
        }
    return out


def band_summary(per_step, bands=STEP_BANDS):
    values = np.asarray(per_step)
    return {f"{lo}-{hi - 1}": float(values[lo:hi].mean()) for lo, hi in bands if lo < len(values)}


def direction_cosines(mean_deltas):
    """Cosine matrix between per-task mean label directions and its mean off-diagonal value."""
    keys = sorted(mean_deltas)
    v = np.stack([np.asarray(mean_deltas[k], dtype=np.float64).reshape(-1) for k in keys])
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    gram = v @ v.T
    off = gram[np.triu_indices(len(keys), 1)]
    return {"tasks": keys, "matrix": gram.tolist(), "mean_off_diagonal": float(off.mean()) if len(off) else float("nan")}


# ---- (b) probes -------------------------------------------------------------------------------------------------------
def ridge_fit(X, y, lam):
    """Ridge classifier on centred X, targets +-1 (dual n x n solve when n <= d, primal otherwise).  Returns (w, b)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    if Xc.shape[0] <= Xc.shape[1]:
        w = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(Xc.shape[0]), yc)
    else:
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    return w, float(y_mean - x_mean @ w)


def mean_direction_fit(X, y):
    """Mean-direction classifier: w = centroid(+1) - centroid(-1), threshold at the midpoint.  Returns (w, b)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    pos, neg = X[y > 0].mean(axis=0), X[y < 0].mean(axis=0)
    w = pos - neg
    return w, float(-0.5 * (pos + neg) @ w)


def group_folds(groups, k=None):
    """(train, test) index pairs: leave-one-group-out, or ``k`` folds of whole groups when k < the number of groups."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chunks = [unique[i::k] for i in range(k)] if k is not None and k < len(unique) else [[g] for g in unique]
    return [(np.flatnonzero(~np.isin(groups, chunk)), np.flatnonzero(np.isin(groups, chunk))) for chunk in chunks]


def ridge_choose_lambda(X, y, groups, lambdas=LAMBDAS, inner_folds=INNER_FOLDS):
    """Lambda with the best inner group-fold accuracy (exact ties -> mean margin, then the larger lambda)."""
    best = None
    for lam in lambdas:
        correct, margin = 0, 0.0
        for train, test in group_folds(groups, inner_folds):
            w, b = ridge_fit(X[train], y[train], lam)
            score = X[test] @ w + b
            correct += int((np.sign(score) == y[test]).sum())
            margin += float((score * y[test]).sum())
        key = (correct, margin, lam)
        if best is None or key > best[0]:
            best = (key, lam)
    return best[1]


def loso_probe(X, y, groups, lambdas=LAMBDAS, outer_folds=OUTER_FOLDS):
    """Held-out-group accuracy of the ridge (nested lambda) and mean-direction classifiers on labels +-1.

    Leave-one-group-out up to ``outer_folds`` groups, ``outer_folds`` folds of whole groups beyond that.  NaN
    accuracies when fewer than two groups or one class are given.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    results = {"n": int(len(y)), "groups": int(len(np.unique(groups))), "chance": float(max((y > 0).mean(), (y < 0).mean()))}
    if results["groups"] < 2 or (y > 0).all() or (y < 0).all():
        for name in ("ridge", "mean_direction"):
            results[name] = {"accuracy": float("nan"), "mean_margin": float("nan")}
        return results
    for name in ("ridge", "mean_direction"):
        correct, chosen, margins = 0, [], []
        for train, test in group_folds(groups, outer_folds):
            if (y[train] > 0).all() or (y[train] < 0).all():
                continue
            if name == "ridge":
                lam = ridge_choose_lambda(X[train], y[train], groups[train], lambdas)
                w, b = ridge_fit(X[train], y[train], lam)
                chosen.append(lam)
            else:
                w, b = mean_direction_fit(X[train], y[train])
            score = X[test] @ w + b
            correct += int((np.sign(score) == y[test]).sum())
            margins.extend((score * y[test] / max(np.linalg.norm(w), 1e-12)).tolist())
        results[name] = {"accuracy": correct / len(y), "mean_margin": float(np.mean(margins))}
        if chosen:
            results[name]["lambdas"] = sorted(set(chosen))
    return results


def probe_rows(file1, file2, pairs):
    """Design matrix of the two files' rows that take part in ``pairs``: X [n, H*A], y (+1 = bowl 2), init-state groups."""
    rows1 = sorted({p[2] for p in pairs})
    rows2 = sorted({p[3] for p in pairs})
    X = np.concatenate([flat(file1["noise"][rows1]), flat(file2["noise"][rows2])])
    y = np.concatenate([-np.ones(len(rows1)), np.ones(len(rows2))])
    groups = np.concatenate([file1["init_state"][rows1], file2["init_state"][rows2]])
    return X, y, groups


# ---- (c) prior ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def prior_mu(prior, feat, instr=None, batch=512):
    """mu [n, H, A] (numpy) of the prior's plain forward on numpy inputs, batched on CPU."""
    out = []
    for i in range(0, len(feat), batch):
        f = torch.as_tensor(np.asarray(feat[i:i + batch], dtype=np.float32))
        e = None if instr is None else torch.as_tensor(np.asarray(instr[i:i + batch], dtype=np.float32))
        mu, _ = prior(f, e)
        out.append(mu.numpy())
    return np.concatenate(out)


def steered_mu(anchor, mu, gain):
    """``VLAPrior.steered`` from precomputed parts: anchor + gain . (mu - anchor) (anchor = mu of the mean instruction)."""
    return anchor + float(gain) * (mu - anchor)


class PriorCache:
    """mu of a concat prior for the rows of a file: the mean-instruction anchor once, each embedding once."""

    def __init__(self, prior):
        self.prior = prior
        self.anchor = {}
        self.mu = {}

    def anchor_mu(self, data, rows):
        key = (data["index"], tuple(rows.tolist()))
        if key not in self.anchor:
            feat = data["feat"][rows]
            mean = np.repeat(self.prior.instr_mean.numpy()[None], len(rows), axis=0)
            self.anchor[key] = prior_mu(self.prior, feat, mean)
        return self.anchor[key]

    def instr_mu(self, data, rows, instr, tag):
        key = (data["index"], tuple(rows.tolist()), tag)
        if key not in self.mu:
            self.mu[key] = prior_mu(self.prior, data["feat"][rows], instr)
        return self.mu[key]

    def steered(self, data, rows, instr, tag, gain):
        mu = self.instr_mu(data, rows, instr, tag)
        return mu if float(gain) == 1.0 else steered_mu(self.anchor_mu(data, rows), mu, gain)


def embeddings_for(file1, file2, pairs, modality):
    """(e_bowl1, e_bowl2) [n_pairs, 768] for the pairs: the bowl file's instr table at the pair's demo (variant 0)."""
    key = f"instr_{modality}"
    if key not in file1 or key not in file2:
        raise KeyError(f"{modality} embeddings missing in {file1['path']} or {file2['path']}")
    e1 = np.stack([file1[key][file1["demo_row"][r1], 0] for _, _, r1, _ in pairs])
    e2 = np.stack([file2[key][file2["demo_row"][r2], 0] for _, _, _, r2 in pairs])
    return e1, e2


def embedding_contrast(prior, e1, e2):
    """Cosine and relative distance of the standardised bowl-1 / bowl-2 embeddings (mean over rows)."""
    mean, std = prior.instr_mean.numpy(), prior.instr_std.numpy()
    z1, z2 = (e1 - mean) / std, (e2 - mean) / std
    cos = (z1 * z2).sum(axis=1) / np.maximum(np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1), 1e-12)
    rel = np.linalg.norm(z2 - z1, axis=1) / np.maximum(0.5 * (np.linalg.norm(z1, axis=1) + np.linalg.norm(z2, axis=1)), 1e-12)
    raw = (e1 * e2).sum(axis=1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-12)
    return {"cos_standardised": float(cos.mean()), "rel_distance_standardised": float(rel.mean()), "cos_raw": float(raw.mean())}


def shift_stats(delta_mu, delta):
    """||delta_mu|| / ||delta||, projection fraction and cosine, averaged over pairs."""
    dm, dl = flat(delta_mu), flat(delta)
    norm_l = np.maximum(np.linalg.norm(dl, axis=1), 1e-12)
    norm_m = np.linalg.norm(dm, axis=1)
    return {"norm_ratio": float((norm_m / norm_l).mean()),
            "projection": float(((dm * dl).sum(axis=1) / norm_l ** 2).mean()),
            "cos": float(((dm * dl).sum(axis=1) / (np.maximum(norm_m, 1e-12) * norm_l)).mean())}


def r_squared(mu, noise):
    noise = flat(noise)
    resid = ((noise - flat(mu)) ** 2).sum()
    total = ((noise - noise.mean(axis=0)) ** 2).sum()
    return float(1.0 - resid / max(total, 1e-12))


def prior_reproduction(prior, file1, file2, pairs, modality, gain, is_val1, is_val2, cache=None):
    """Part (c) for one prior / modality / gain on the pairs (rows of both files)."""
    r1 = np.array([p[2] for p in pairs])
    r2 = np.array([p[3] for p in pairs])
    delta = (file2["noise"][r2] - file1["noise"][r1]).astype(np.float64)
    feat1, feat2 = file1["feat"][r1], file2["feat"][r2]
    out = {"pairs": len(pairs)}
    if prior.instr_mode == "concat":
        cache = cache or PriorCache(prior)
        e1, e2 = embeddings_for(file1, file2, pairs, modality)
        mu1_e1 = cache.steered(file1, r1, e1, (modality, 1), gain)
        mu1_e2 = cache.steered(file1, r1, e2, (modality, 2), gain)
        mu2_e1 = cache.steered(file2, r2, e1, (modality, 1), gain)
        mu2_e2 = cache.steered(file2, r2, e2, (modality, 2), gain)
        delta_mu = 0.5 * ((mu1_e2 - mu1_e1) + (mu2_e2 - mu2_e1))
        out.update(shift_stats(delta_mu, delta))
        out["shift_row1"] = shift_stats(mu1_e2 - mu1_e1, delta)
        # Pairs with a held-out row vs pairs of training rows only: is the shortfall the same on unseen states?
        pair_val = is_val1[r1] | is_val2[r2]
        for tag, keep in (("train", ~pair_val), ("val", pair_val)):
            out[f"shift_{tag}"] = dict(shift_stats(delta_mu[keep], delta[keep]), pairs=int(keep.sum())) if keep.any() else None
        out["delta_mu_norm"] = float(np.linalg.norm(flat(delta_mu), axis=1).mean())
        own1, own2 = mu1_e1, mu2_e2
        out["embedding"] = embedding_contrast(prior, e1, e2)
        # Offline bowl choice: the labels' mean-direction classifier applied to mu of the shared (bowl-1 row) observation.
        X, y, _ = probe_rows(file1, file2, pairs)
        w, b = mean_direction_fit(X, y)
        out["mu_says_bowl2"] = {"instr_bowl2": float(((flat(mu1_e2) @ w + b) > 0).mean()),
                                "instr_bowl1": float(((flat(mu1_e1) @ w + b) > 0).mean())}
    else:
        own1, own2 = prior_mu(prior, feat1), prior_mu(prior, feat2)
        out.update(shift_stats(own2 - own1, delta))
        out["delta_mu_norm"] = float(np.linalg.norm(flat(own2 - own1), axis=1).mean())
    noise = np.concatenate([file1["noise"][r1], file2["noise"][r2]])
    mu_own = np.concatenate([own1, own2])
    val = np.concatenate([is_val1[r1], is_val2[r2]])
    out["r2_all"] = r_squared(mu_own, noise)
    out["r2_train"] = r_squared(mu_own[~val], noise[~val]) if (~val).sum() > 1 else float("nan")
    out["r2_val"] = r_squared(mu_own[val], noise[val]) if val.sum() > 1 else float("nan")
    out["val_rows"] = int(val.sum())
    return out


# ---- driver ----------------------------------------------------------------------------------------------------------------
def parse_list(spec, cast=float):
    return [cast(x) for x in str(spec).split(",") if x.strip()]


def parse_int_spec(spec):
    """``"0-9"``, ``"0,3,5"`` -> sorted ints."""
    ids = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prior", type=Path, nargs="*", default=[], help="VLAPrior checkpoints (concat and/or none)")
    parser.add_argument("--tasks", default=None, help="suite task ids (default: every task with both bowl files)")
    parser.add_argument("--modalities", default="text,sketch,audio")
    parser.add_argument("--gains", default="1,2,3")
    parser.add_argument("--max-query", type=int, default=10, help="pairs at query indices 0..max-query")
    parser.add_argument("--min-states", type=int, default=MIN_STATES, help="skip tasks with fewer paired init states")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", default="demo", help="trainer split kind (demo, or episode = init states per task)")
    parser.add_argument("--val-init-states", default=None, help="held-out init states (overrides the trainer split)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    args.gains = parse_list(args.gains, float)
    args.task_ids = parse_int_spec(args.tasks) if args.tasks else None
    args.val_states = parse_int_spec(args.val_init_states) if args.val_init_states else None
    return args


def analyse(files, task_ids, max_query, val, priors, modalities, gains, min_states=MIN_STATES):
    """All three parts for the given tasks; returns a JSON-serialisable dict.

    Tasks with fewer than ``min_states`` paired init states are listed with their pair count and skipped (a held-out
    state needs at least two others to learn from).
    """
    report = {"tasks": {}, "pooled": {}, "max_query": max_query}
    pooled_q0, pooled_qn, mean_deltas = [], [], {}
    caches = {name: PriorCache(prior) for name, prior in priors.items()}
    for t in task_ids:
        f1, f2 = files[100 + t], files[200 + t]
        pairs = matched_pairs(f1, f2, max_query)
        states = sorted({p[0] for p in pairs})
        if len(states) < min_states:
            report["tasks"][str(t)] = {"pairs": 0, "paired_init_states": len(states),
                                       "skipped": f"fewer than {min_states} paired init states"}
            continue
        q0 = [p for p in pairs if p[1] == 0]
        task = {"pairs": len(pairs), "pairs_q0": len(q0), "init_states": sorted({p[0] for p in pairs}),
                "labels": {str(q): v for q, v in label_shift(f1, f2, pairs, queries=(0, max_query)).items()}}
        for q, stats in task["labels"].items():
            mean_deltas[(t, q)] = stats.pop("mean_delta")
            stats["per_step_bands"] = band_summary(stats["per_step"])
        shift_all = label_shift(f1, f2, pairs, queries=sorted({p[1] for p in pairs}))
        task["labels"]["mean_over_queries"] = {k: float(np.mean([s[k] for s in shift_all.values()]))
                                               for k in ("mean_abs_delta_over_sigma", "d_prime", "pairwise_cos", "shared_fraction")}
        X0, y0, g0 = probe_rows(f1, f2, q0)
        Xn, yn, gn = probe_rows(f1, f2, pairs)
        task["probe"] = {"q0": loso_probe(X0, y0, g0), "all_queries": loso_probe(Xn, yn, gn)}
        pooled_q0.append((X0, y0, g0 + 1000 * t))
        pooled_qn.append((Xn, yn, gn + 1000 * t))
        task["priors"] = {}
        for name, prior in priors.items():
            entry = {}
            if prior.instr_mode == "concat":
                for modality in modalities:
                    if f"instr_{modality}" not in f1 or f"instr_{modality}" not in f2:
                        continue
                    for gain in gains:
                        entry[f"{modality}/g{gain:g}"] = {
                            "q0": prior_reproduction(prior, f1, f2, q0, modality, gain, val[100 + t], val[200 + t], caches[name]),
                            "all_queries": prior_reproduction(prior, f1, f2, pairs, modality, gain, val[100 + t], val[200 + t],
                                                              caches[name])}
            else:
                entry["none"] = {"q0": prior_reproduction(prior, f1, f2, q0, None, 1.0, val[100 + t], val[200 + t]),
                                 "all_queries": prior_reproduction(prior, f1, f2, pairs, None, 1.0, val[100 + t], val[200 + t])}
            task["priors"][name] = entry
        report["tasks"][str(t)] = task
    if pooled_q0:
        report["pooled"]["probe_q0"] = loso_probe(*[np.concatenate(x) for x in zip(*pooled_q0)])
        report["pooled"]["probe_all_queries"] = loso_probe(*[np.concatenate(x) for x in zip(*pooled_qn)])
        q0_dirs = {t: d for (t, q), d in mean_deltas.items() if q == "0"}
        report["pooled"]["direction_cos_q0"] = direction_cosines(q0_dirs)
    return report


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown(report, priors):
    lines = []
    tasks = [t for t in report["tasks"] if report["tasks"][t].get("pairs")]
    lines.append("### (a) label shift delta = noise(bowl 2) - noise(bowl 1), matched init states (q0 = first query; "
                 f"q<= = queries 0..{report['max_query']} averaged)")
    lines.append("| task | states | ‖δ‖ q0 | within-bowl pair ‖·‖ | mean\\|δ\\|/σ q0 | d' q0 | pairwise cos q0 | shared frac q0 | "
                 "mean\\|δ\\|/σ q<= | d' q<= | cos q<= | steps 0-9/10-19/20-29/30-39/40-49 (q0) | dims dx dy dz rx ry rz g (q0) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        task = report["tasks"][t]
        l0, lm = task["labels"]["0"], task["labels"]["mean_over_queries"]
        bands = "/".join(fmt(v, 1) for v in l0["per_step_bands"].values())
        dims = " ".join(fmt(v, 1) for v in l0["per_dim"])
        lines.append(f"| {t} | {l0['init_states']} | {fmt(l0['delta_norm'], 1)} | {fmt(l0['within_pair_norm'], 1)} | "
                     f"{fmt(l0['mean_abs_delta_over_sigma'])} | {fmt(l0['d_prime'])} | {fmt(l0['pairwise_cos'])} | "
                     f"{fmt(l0['shared_fraction'])} | {fmt(lm['mean_abs_delta_over_sigma'])} | {fmt(lm['d_prime'])} | "
                     f"{fmt(lm['pairwise_cos'])} | {bands} | {dims} |")
    if "direction_cos_q0" in report["pooled"]:
        lines.append(f"\nMean cosine between the tasks' mean q0 directions: {fmt(report['pooled']['direction_cos_q0']['mean_off_diagonal'])}")
    lines.append("\n### (b) leave-init-state-out bowl probes on the noise labels (accuracy; chance = majority class)")
    lines.append("| task | rows q0 | ridge q0 | mean-dir q0 | rows q<= | ridge q<= | mean-dir q<= | chance |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in tasks:
        p = report["tasks"][t]["probe"]
        lines.append(f"| {t} | {p['q0']['n']} | {fmt(p['q0']['ridge']['accuracy'])} | {fmt(p['q0']['mean_direction']['accuracy'])} | "
                     f"{p['all_queries']['n']} | {fmt(p['all_queries']['ridge']['accuracy'])} | "
                     f"{fmt(p['all_queries']['mean_direction']['accuracy'])} | {fmt(p['q0']['chance'])} |")
    if "probe_q0" in report["pooled"]:
        p0, pn = report["pooled"]["probe_q0"], report["pooled"]["probe_all_queries"]
        lines.append(f"| pooled | {p0['n']} | {fmt(p0['ridge']['accuracy'])} | {fmt(p0['mean_direction']['accuracy'])} | "
                     f"{pn['n']} | {fmt(pn['ridge']['accuracy'])} | {fmt(pn['mean_direction']['accuracy'])} | {fmt(p0['chance'])} |")
    for name in priors:
        lines.append(f"\n### (c) prior `{name}`: delta_mu = mu(e_bowl2) - mu(e_bowl1) against the label shift "
                     "(q0 rows; proj = reproduced fraction along delta; mu->b2 = labels' mean-direction classifier on mu)")
        lines.append("| task | modality/gain | ‖δμ‖/‖δ‖ | proj | cos | proj train / val (pairs) | mu->b2 \\| e_b2 | mu->b2 \\| e_b1 | "
                     "R² train / val (own instr) | proj q<= | cos(e_b1, e_b2) std | ‖Δe‖/‖e‖ std |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in tasks:
            for key, cell in report["tasks"][t]["priors"].get(name, {}).items():
                c0, cn = cell["q0"], cell["all_queries"]
                says = c0.get("mu_says_bowl2", {})
                emb = c0.get("embedding", {})
                tv = [c0.get(f"shift_{k}") for k in ("train", "val")]
                split = " / ".join("n/a" if x is None else f"{fmt(x['projection'])} ({x['pairs']})" for x in tv)
                lines.append(f"| {t} | {key} | {fmt(c0['norm_ratio'])} | {fmt(c0['projection'])} | {fmt(c0['cos'])} | {split} | "
                             f"{fmt(says.get('instr_bowl2'))} | {fmt(says.get('instr_bowl1'))} | "
                             f"{fmt(c0['r2_train'])} / {fmt(c0['r2_val'])} | {fmt(cn['projection'])} | "
                             f"{fmt(emb.get('cos_standardised'), 3)} | {fmt(emb.get('rel_distance_standardised'), 3)} |")
    return "\n".join(lines)


def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main(argv=None):
    args = parse_args(argv)
    files = load_cf_dataset(args.data, MODALITIES)
    task_ids = args.task_ids or paired_tasks(files)
    missing = [t for t in task_ids if 100 + t not in files or 200 + t not in files]
    if missing:
        raise SystemExit(f"tasks {missing} lack a bowl-1 or bowl-2 file under {args.data}")
    val = val_masks(files, args.val_fraction, args.seed, args.val_states, args.val_split)
    priors = {}
    for path in args.prior:
        prior, _ = VLAPrior.load(path)
        priors[Path(path).parent.name if Path(path).name in ("best.pt", "prior.pt") else Path(path).stem] = prior
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    report = analyse(files, task_ids, args.max_query, val, priors, args.modalities, args.gains, args.min_states)
    report["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report["args"]["prior"] = [str(p) for p in args.prior]
    text = markdown(report, priors)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_json(report), indent=1))
        args.out.with_suffix(".md").write_text(text + "\n")
        print(f"\nwrote {args.out} and {args.out.with_suffix('.md')}")
    return report


if __name__ == "__main__":
    main()
