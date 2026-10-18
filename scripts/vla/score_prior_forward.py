#!/usr/bin/env python
"""Offline steering score of a VLA prior: forward the frozen flow from the prior's x0 and classify the first chunk.

No simulator.  For every language-sensitive task and init state with a successful bowl-1 AND bowl-2 rollout
(``data/vla/counterfactual``), the observation of the FIRST query (env step 0; both rollouts start from the same
state, the bowl-1 file's frame is used) is rebuilt under the NEUTRAL prompt exactly as ``reverse_rollouts.py`` does
(policy-view images rotated back by ``counterfactual.raw_images``, raw 8-d state; ``pi05_flow.prepare_batches``
flips and normalises).  Source noises x0 are then pushed through the flow (``forward_from_noise``, --num-steps Euler
steps, as inference) and the generated 50-step chunk is classified as bowl 1 / bowl 2:

- nearest   the recorded first chunk of the two rollouts of that init state that is closer in the normalised
            action space (margin = (d1 - d2) / ||c1 - c2||, positive = bowl 2);
- geometric the cumulative xy displacement of the first --horizon steps (un-normalised actions are world-frame eef
            deltas) compared with the directions from the gripper (state[:3]) to the two bowls (``positions``).

Conditions: the stored reversal labels of both rollouts (``--data``; sanity: each must regenerate its own chunk),
Gaussian x0 (--samples draws), and for every --prior x modality x gain x instructed bowl, --samples draws of
x0 = mu + sigma_scale . sigma . eps with mu from ``VLAPrior.steered(feat, e_bowl, gain)`` (e_bowl = the dataset's
stored text / sketch / audio embedding of the (task, bowl, init state)).  Reported per task and overall:
P(bowl 2 | bowl-2 instruction) and P(bowl 2 | bowl-1 instruction), both classifiers, plus the mean margins.

Init states: the rollouts exist for 0-19 (repeat 0); 20-39 are the simulator's held-out states and have no rollouts,
so --init-states names the states to score (a training-time hold-out of some of 0-19 is the way to get held-out
scores).  Output: JSON at --out (+ a markdown table next to it and on stdout).

Usage (remote, GPU 1, VLA venv; ~5 min per prior with --modalities text --gains 1 --samples 8):
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python scripts/vla/score_prior_forward.py \\
        --rollouts data/vla/counterfactual --data data/vla/libero_spatial_counterfactual \\
        --prior outputs/vla_cf/prior-cf-concat/best.pt --modalities text --gains 1,2,3 --samples 8 \\
        --init-states 0-19 --out outputs/vla_cf/score_forward/prior-cf-concat.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leap.vla import counterfactual as C  # noqa: E402
from leap.vla import libero_scene as S  # noqa: E402
from leap.vla.prior import VLAPrior  # noqa: E402

MODALITIES = ("text", "sketch", "audio")
DEFAULT_TASKS = "1,3,5,6,7,9"
INIT_STATES_PER_REPEAT = int(getattr(C, "INIT_STATES_PER_REPEAT", 100))
REFERENCE_CONDITIONS = ("stored-b1", "stored-b2", "gauss")


def flow_api():
    """``leap.vla.pi05_flow`` (imported lazily; tests monkeypatch this)."""
    from leap.vla import pi05_flow
    return pi05_flow


# ---- arguments -------------------------------------------------------------------------------------------------
def parse_spec(spec):
    """``"0-19"``, ``"0,3,5"`` -> sorted ints."""
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
    if not ids:
        raise ValueError(f"no ids in {spec!r}")
    return sorted(ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, default=Path("data/vla/counterfactual"))
    parser.add_argument("--data", type=Path, default=Path("data/vla/libero_spatial_counterfactual"),
                        help="counterfactual reversal dataset (instruction embeddings and stored labels)")
    parser.add_argument("--prior", type=Path, nargs="*", default=[], help="VLAPrior checkpoints to score")
    parser.add_argument("--modalities", default="text")
    parser.add_argument("--gains", default="1")
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=8, help="x0 draws per (prior, modality, gain, bowl) and for gauss")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--init-states", default="0-19")
    parser.add_argument("--repeat", type=int, default=0, help="rollout repeat whose files give the observation and reference chunks")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=20, help="chunk steps summed by the geometric classifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-references", action="store_true", help="skip the stored-label and Gaussian conditions")
    parser.add_argument("--allow-lifted-mismatch", action="store_true",
                        help="keep rollouts whose first lifted bowl differs from the prompted one (default: require both)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.batch_size < 1 or args.num_steps < 1 or args.horizon < 1:
        parser.error("samples, batch-size, num-steps and horizon must be positive")
    if args.sigma_scale < 0:
        parser.error("sigma-scale must be >= 0")
    args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    unknown = [m for m in args.modalities if m not in MODALITIES]
    if unknown:
        parser.error(f"unknown modalities {unknown}; expected a subset of {MODALITIES}")
    args.gains = [float(g) for g in args.gains.split(",") if g.strip()]
    try:
        args.task_ids = parse_spec(args.tasks)
        args.states = parse_spec(args.init_states)
    except ValueError as error:
        parser.error(str(error))
    return args


# ---- data ------------------------------------------------------------------------------------------------------
def episode_of(repeat, init_state):
    return int(repeat) * INIT_STATES_PER_REPEAT + int(init_state)


def select_states(rollouts, task, init_states, repeat=0, require_lifted=True):
    """``([(init_state, path_bowl1, path_bowl2)], {init_state: reason})``: states with both rollouts usable."""
    chosen, skipped = [], {}
    for s in init_states:
        episode = episode_of(repeat, s)
        paths = {b: C.episode_path(rollouts, task, b, episode) for b in (1, 2)}
        if not all(p.exists() for p in paths.values()):
            skipped[s] = "missing " + ",".join(f"bowl{b}" for b, p in paths.items() if not p.exists())
            continue
        scalars = {b: C.episode_scalars(p) for b, p in paths.items()}
        bad = []
        for b, sc in scalars.items():
            if not bool(sc.get("success", False)):
                bad.append(f"bowl{b} failed")
            elif require_lifted and int(sc.get("lifted_bowl", b)) != b:
                bad.append(f"bowl{b} lifted bowl {sc.get('lifted_bowl')} first")
        if bad:
            skipped[s] = "; ".join(bad)
            continue
        chosen.append((s, paths[1], paths[2]))
    return chosen, skipped


def first_query(data):
    """Row of the recorded query made at env step 0."""
    rows = np.flatnonzero(np.asarray(data["pred_step"]) == 0)
    if not len(rows):
        raise ValueError("episode has no query at env step 0")
    return int(rows[0])


def load_dataset_files(data_dir, tasks):
    """``{index: {noise, demo_id, t, instr_*}}`` of the bowl files of ``tasks`` that exist (feat is not read)."""
    files = {}
    for t in tasks:
        for b in (1, 2):
            path = Path(data_dir) / f"task_{100 * b + t}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as npz:
                entry = {"noise": np.asarray(npz["noise"], dtype=np.float32), "demo_id": np.asarray(npz["demo_id"]),
                         "t": np.asarray(npz["t"])}
                for m in MODALITIES:
                    key = f"instr_{m}"
                    if key in npz and np.any(npz[key]):
                        entry[key] = np.asarray(npz[key], dtype=np.float32)
            entry["demos"] = np.unique(entry["demo_id"])
            files[100 * b + t] = entry
    return files


def stored_label(files, task, bowl, episode):
    """The reversal noise [H, A] of the first query (t == 0) of the rollout, or None when the dataset lacks it."""
    entry = files.get(100 * bowl + task)
    if entry is None:
        return None
    rows = np.flatnonzero((entry["demo_id"] == C.demo_id_for(bowl, episode)) & (entry["t"] == 0))
    return None if not len(rows) else entry["noise"][rows[0]]


def instruction_embedding(files, task, bowl, episode, modality):
    """The stored embedding [768] of (task, bowl) for the rollout's demo (sketch: that demo; text/audio: any row)."""
    entry = files.get(100 * bowl + task)
    key = f"instr_{modality}"
    if entry is None or key not in entry:
        raise KeyError(f"no {modality} embeddings for task {task} bowl {bowl} under the dataset")
    demo = C.demo_id_for(bowl, episode)
    where = np.flatnonzero(entry["demos"] == demo)
    if len(where):
        return entry[key][int(where[0]), 0]
    if modality == "sketch":
        return None
    return entry[key][0, 0]


# ---- classifiers ------------------------------------------------------------------------------------------------
def nearest_chunk(actions_norm, chunk1_norm, chunk2_norm):
    """Bowl (1 or 2) of each chunk by the nearer recorded chunk and the margin (d1 - d2) / ||c1 - c2|| (> 0: bowl 2)."""
    x = np.asarray(actions_norm, dtype=np.float64).reshape(len(actions_norm), -1)
    c1 = np.asarray(chunk1_norm, dtype=np.float64).reshape(-1)
    c2 = np.asarray(chunk2_norm, dtype=np.float64).reshape(-1)
    d1 = np.linalg.norm(x - c1, axis=1)
    d2 = np.linalg.norm(x - c2, axis=1)
    margin = (d1 - d2) / max(np.linalg.norm(c1 - c2), 1e-12)
    return np.where(margin > 0, 2, 1), margin


def geometric_bowl(actions, eef_xyz, bowl1_xyz, bowl2_xyz, horizon=20):
    """Bowl by the direction of the cumulative xy action over ``horizon`` steps; margin = cos(to bowl 2) - cos(to bowl 1)."""
    actions = np.asarray(actions, dtype=np.float64)
    v = actions[:, :horizon, :2].sum(axis=1)
    eef = np.asarray(eef_xyz, dtype=np.float64)[:2]
    d1 = np.asarray(bowl1_xyz, dtype=np.float64)[:2] - eef
    d2 = np.asarray(bowl2_xyz, dtype=np.float64)[:2] - eef
    norm_v = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    cos1 = v @ d1 / (norm_v * max(np.linalg.norm(d1), 1e-12))
    cos2 = v @ d2 / (norm_v * max(np.linalg.norm(d2), 1e-12))
    margin = cos2 - cos1
    return np.where(margin > 0, 2, 1), margin


# ---- scoring -------------------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, args):
        self.args = args
        self.flow = flow_api()
        self.device = torch.device(args.device)
        self.bundle = self.flow.load_pi05(args.checkpoint, args.device)
        self.priors = {}
        for path in args.prior:
            prior, _ = VLAPrior.load(path, map_location=args.device)
            name = Path(path).parent.name if Path(path).name in ("best.pt", "prior.pt") else Path(path).stem
            self.priors[name] = prior.to(self.device)
        self.files = load_dataset_files(args.data, args.task_ids)
        self.generator = torch.Generator(device=self.device)

    def prefix(self, images, state):
        """Cached prefix of ``batch_size`` copies of one observation under the neutral prompt, and its features [1, F]."""
        obs = self.flow.Observation(images=images, state=np.asarray(state, dtype=np.float32), prompt=S.NEUTRAL_PROMPT)
        batch = self.flow.prepare_batches(self.bundle, [obs] * self.args.batch_size)
        ctx = self.flow.prefix_context(self.bundle, batch)
        feat = torch.as_tensor(self.flow.prefix_features(self.bundle, batch, ctx)).float()[:1].to(self.device)
        return ctx, feat

    def forward(self, ctx, noises):
        """``forward_from_noise`` on [n, H, 32] noises in batches of the prefix's size; returns normalised [n, H, 32]."""
        n, b = noises.shape[0], self.args.batch_size
        out = []
        for start in range(0, n, b):
            chunk = noises[start:start + b]
            if chunk.shape[0] < b:
                chunk = torch.cat([chunk, chunk[-1:].expand(b - chunk.shape[0], -1, -1)])
            result = self.flow.forward_from_noise(self.bundle, ctx, chunk.to(self.device), self.args.num_steps)
            out.append(torch.as_tensor(result).float()[:min(b, n - start)])
        return torch.cat(out)

    def noise_sets(self, feat, task, episode):
        """``{condition: [n, H, 32] noise}`` for one observation (stored labels, Gaussian, prior samples)."""
        args, pad = self.args, self.flow.pad_noise
        sets, notes = {}, {}
        chunk_size = int(self.bundle.config.chunk_size)
        max_dim = int(self.bundle.config.max_action_dim)
        if not args.no_references:
            for b in (1, 2):
                label = stored_label(self.files, task, b, episode)
                if label is None:
                    notes[f"stored-b{b}"] = "no stored label"
                else:
                    sets[f"stored-b{b}"] = pad(torch.as_tensor(label)[None].to(self.device), max_dim, generator=self.generator)
            sets["gauss"] = torch.randn(args.samples, chunk_size, max_dim, device=self.device, generator=self.generator)
        for name, prior in self.priors.items():
            modalities = args.modalities if prior.instr_mode == "concat" else [None]
            gains = args.gains if prior.instr_mode == "concat" else [1.0]
            for modality in modalities:
                for gain in gains:
                    for b in ((1, 2) if modality else (None,)):
                        condition = f"{name}/{modality}/g{gain:g}/instr-b{b}" if modality else f"{name}/none"
                        instr = None
                        if modality:
                            vector = instruction_embedding(self.files, task, b, episode, modality)
                            if vector is None:
                                notes[condition] = "no sketch for this demo"
                                continue
                            instr = torch.as_tensor(vector)[None].to(self.device)
                        mu, log_var = prior.steered(feat, instr, gain)
                        mu = mu.expand(args.samples, -1, -1)
                        log_var = log_var.expand(args.samples, -1, -1)
                        x0 = prior.sample(mu, log_var, args.sigma_scale, generator=self.generator)
                        sets[condition] = pad(x0, max_dim, generator=self.generator)
        return sets, notes

    def score_state(self, task, init_state, path1, path2):
        """Classify every condition's chunks for one init state; returns {condition: {...}} plus the reference info."""
        args = self.args
        data1, data2 = C.load_episode(path1), C.load_episode(path2)
        episode = int(data1.get("episode", episode_of(args.repeat, init_state)))
        q1, q2 = first_query(data1), first_query(data2)
        state = np.asarray(data1["state"][0], dtype=np.float32)
        state_gap = float(np.abs(state - np.asarray(data2["state"][0], dtype=np.float32)).max())
        images = C.raw_images(data1["agentview"][0], data1["wrist"][0])
        positions = np.asarray(data1["positions"][0], dtype=np.float64)
        chunks = np.stack([data1["chunk"][q1], data2["chunk"][q2]]).astype(np.float32)
        self.generator.manual_seed(args.seed * 1_000_000 + task * 1_000 + init_state)
        ctx, feat = self.prefix(images, state)
        sets, notes = self.noise_sets(feat, task, episode)
        names = list(sets)
        noises = torch.cat([sets[k] for k in names])
        actions_norm = self.forward(ctx, noises)[:, :, :chunks.shape[-1]].cpu().numpy()
        actions = self.flow.unnormalize_actions(self.bundle, torch.as_tensor(self.forward_pad(actions_norm))).cpu().numpy()
        ref_norm = torch.as_tensor(self.flow.normalize_actions(self.bundle, chunks)).float().cpu().numpy()[:, :, :chunks.shape[-1]]
        bowl_near, margin_near = nearest_chunk(actions_norm, ref_norm[0], ref_norm[1])
        bowl_geo, margin_geo = geometric_bowl(actions, state[:3], positions[0], positions[1], args.horizon)
        ref_geo, _ = geometric_bowl(chunks, state[:3], positions[0], positions[1], args.horizon)
        results, start = {}, 0
        for name in names:
            n = sets[name].shape[0]
            sl = slice(start, start + n)
            results[name] = {"n": n, "nearest_b2": int((bowl_near[sl] == 2).sum()), "geometric_b2": int((bowl_geo[sl] == 2).sum()),
                             "margin_nearest": float(margin_near[sl].mean()), "margin_geometric": float(margin_geo[sl].mean())}
            start += n
        info = {"episode": episode, "state_gap": state_gap, "recorded_geometric": ref_geo.tolist(), "notes": notes,
                "recon_err": {k: float(np.linalg.norm(actions_norm[names.index(k)] - ref_norm[b - 1]) / np.linalg.norm(ref_norm[b - 1]))
                              for b, k in ((1, "stored-b1"), (2, "stored-b2")) if k in sets}}
        return results, info

    def forward_pad(self, actions_norm7):
        """[n, H, 7] -> [n, H, 32] zero-padded (unnormalize_actions drops the padding again)."""
        max_dim = int(self.bundle.config.max_action_dim)
        out = np.zeros(actions_norm7.shape[:2] + (max_dim,), dtype=np.float32)
        out[:, :, :actions_norm7.shape[-1]] = actions_norm7
        return out


def aggregate(per_state):
    """``{condition: {"tasks": {task: stats}, "overall": stats}}`` from ``{task: {init_state: {condition: result}}}``."""
    conditions = {}
    for task, states in per_state.items():
        for s, results in states.items():
            for name, r in results.items():
                conditions.setdefault(name, {}).setdefault(task, []).append(r)
    summary = {}
    for name, by_task in conditions.items():
        entry = {"tasks": {}, "overall": None}
        pooled = []
        for task, rows in sorted(by_task.items()):
            entry["tasks"][str(task)] = stats_of(rows)
            pooled += rows
        entry["overall"] = stats_of(pooled)
        summary[name] = entry
    return summary


def stats_of(rows):
    n = sum(r["n"] for r in rows)
    return {"states": len(rows), "samples": n,
            "p_bowl2_nearest": sum(r["nearest_b2"] for r in rows) / max(n, 1),
            "p_bowl2_geometric": sum(r["geometric_b2"] for r in rows) / max(n, 1),
            "margin_nearest": float(np.mean([r["margin_nearest"] for r in rows])) if rows else float("nan"),
            "margin_geometric": float(np.mean([r["margin_geometric"] for r in rows])) if rows else float("nan")}


def markdown(summary, tasks):
    lines = ["| condition | " + " | ".join(f"task {t}" for t in tasks) + " | overall nearest | overall geometric | margin near / geo |",
             "|---|" + "---|" * len(tasks) + "---|---|---|"]
    for name, entry in summary.items():
        cells = []
        for t in tasks:
            st = entry["tasks"].get(str(t))
            cells.append("n/a" if st is None else f"{100 * st['p_bowl2_nearest']:.0f} / {100 * st['p_bowl2_geometric']:.0f}")
        o = entry["overall"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {100 * o['p_bowl2_nearest']:.1f} % | {100 * o['p_bowl2_geometric']:.1f} % | "
                     f"{o['margin_nearest']:+.3f} / {o['margin_geometric']:+.3f} |")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    scorer = Scorer(args)
    per_state, skipped, infos = {}, {}, {}
    for task in args.task_ids:
        chosen, skip = select_states(args.rollouts, task, args.states, args.repeat, not args.allow_lifted_mismatch)
        skipped[str(task)] = {str(s): reason for s, reason in skip.items()}
        per_state[task] = {}
        with torch.inference_mode():
            for s, path1, path2 in chosen:
                results, info = scorer.score_state(task, s, path1, path2)
                per_state[task][s] = results
                infos[f"{task}:{s}"] = info
                print(f"[task {task} state {s}] " + ", ".join(f"{k}: {v['nearest_b2']}/{v['n']}" for k, v in results.items())
                      + f" ({time.time() - started:.0f}s)", flush=True)
    summary = aggregate(per_state)
    tasks_scored = [t for t in args.task_ids if per_state.get(t)]
    text = markdown(summary, tasks_scored)
    print("\nP(bowl 2) in % per task (nearest / geometric) and overall:\n" + text)
    gaps = [i["state_gap"] for i in infos.values()]
    recon = [v for i in infos.values() for v in i["recon_err"].values()]
    print(f"\nstates scored: {sum(len(v) for v in per_state.values())}, skipped: {sum(len(v) for v in skipped.values())}; "
          f"max state gap between the two rollouts {max(gaps) if gaps else float('nan'):.2e}; "
          f"stored-label reconstruction error mean {np.mean(recon) if recon else float('nan'):.3f}; {time.time() - started:.0f}s")
    report = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "summary": summary, "per_state": {str(t): {str(s): r for s, r in v.items()} for t, v in per_state.items()},
              "skipped": skipped, "info": infos, "seconds": time.time() - started}
    report["args"]["prior"] = [str(p) for p in args.prior]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        args.out.with_suffix(".md").write_text(text + "\n")
        print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
