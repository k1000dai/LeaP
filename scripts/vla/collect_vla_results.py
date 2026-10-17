"""Aggregate the summary.json files written by scripts/vla/eval_vla_steer.py into one table.

Scans <out-dir>/*/summary.json (one run per directory, e.g. outputs/vla/neutral-prior-sketch-s1),
prints a Markdown table with one row per condition/modality/sigma (success rate, bowl-choice
accuracy, episodes) and a per-task success table, and writes results.json, results_per_task.csv
and results.md next to them.  Prior training runs (<out-dir>/prior-*/run.json) are listed with
their validation NLL and the reference NLLs when present.

Example:
    python scripts/vla/collect_vla_results.py --out-dir outputs/vla
"""
import argparse
import csv
import json
from pathlib import Path

CONDITION_ORDER = {"full-prompt": 0, "neutral-gauss": 1, "neutral-prior": 2}
MODALITY_ORDER = {"none": 0, "text": 1, "sketch": 2, "audio": 3}


def percent(value):
    return "" if value is None or value != value else f"{100 * value:.1f}"


def load_runs(out_dir):
    """One dict per summary.json, sorted by condition, modality and sigma."""
    runs = []
    for path in sorted(Path(out_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        overall = summary.get("overall", {})
        runs.append({"run": path.parent.name, "condition": summary.get("condition"),
                     "modality": summary.get("modality", "none"), "sigma_scale": summary.get("sigma_scale"),
                     "seed": summary.get("seed"), "episodes": overall.get("episodes", 0),
                     "success_rate": overall.get("success_rate"),
                     "bowl_choice_accuracy": overall.get("bowl_choice_accuracy"),
                     "bowl_choice_counts": overall.get("bowl_choice_counts", {}),
                     "mean_steps": overall.get("mean_steps"), "prior": summary.get("prior"),
                     "tasks": {task: {"success_rate": row.get("success_rate"), "episodes": row.get("episodes"),
                                      "bowl_choice_accuracy": row.get("bowl_choice_accuracy"),
                                      "task_language": row.get("task_language")}
                               for task, row in summary.get("tasks", {}).items()},
                     "path": str(path)})
    runs.sort(key=lambda r: (CONDITION_ORDER.get(r["condition"], 9), MODALITY_ORDER.get(r["modality"], 9),
                             -(r["sigma_scale"] if r["sigma_scale"] is not None else 0), r["run"]))
    return runs


def load_priors(out_dir):
    priors = []
    for path in sorted(Path(out_dir).glob("prior-*/run.json")):
        run = json.loads(path.read_text())
        references = run.get("references", {})
        final = run.get("final", {})
        priors.append({"run": path.parent.name, "instr_mode": run.get("args", {}).get("instr_mode"),
                       "modalities": run.get("args", {}).get("modalities"), "steps": run.get("steps"),
                       "n_train": run.get("n_train"), "n_val": run.get("n_val"),
                       "best_val_nll": run.get("best", {}).get("val_nll"), "final_val_nll": final.get("val_nll"),
                       "val_nll_by_modality": {k[len("val_nll_"):]: v for k, v in final.items()
                                               if k.startswith("val_nll_")},
                       "gauss_val_nll": references.get("gauss_val_nll"),
                       "empirical_val_nll": references.get("empirical_val_nll"),
                       "recon_err": run.get("recon_err"), "path": str(path)})
    return priors


def markdown(runs, priors):
    lines = ["| run | condition | modality | sigma | episodes | success % | bowl-choice % | target/distractor/none |",
             "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        counts = run["bowl_choice_counts"]
        sigma = "" if run["condition"] != "neutral-prior" or run["sigma_scale"] is None else f"{run['sigma_scale']:g}"
        lines.append(f"| {run['run']} | {run['condition']} | {run['modality']} | {sigma} | {run['episodes']} | "
                     f"{percent(run['success_rate'])} | {percent(run['bowl_choice_accuracy'])} | "
                     f"{counts.get('target', 0)}/{counts.get('distractor', 0)}/{counts.get('none', 0)} |")
    tasks = sorted({task for run in runs for task in run["tasks"]}, key=int)
    if tasks:
        lines += ["", "| run | " + " | ".join(f"task {t}" for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
        for run in runs:
            lines.append(f"| {run['run']} | " + " | ".join(
                percent(run["tasks"][t]["success_rate"]) if t in run["tasks"] else "" for t in tasks) + " |")
    if priors:
        lines += ["", "| prior | instr_mode | steps | train/val | best val NLL | per modality | N(0,I) NLL | empirical NLL | recon err (median) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for prior in priors:
            def fmt(value):
                return "" if value is None else f"{value:.3f}"
            per_modality = ", ".join(f"{k} {v:.3f}" for k, v in prior["val_nll_by_modality"].items())
            recon = (prior.get("recon_err") or {}).get("median")
            lines.append(f"| {prior['run']} | {prior['instr_mode']} | {prior['steps']} | "
                         f"{prior['n_train']}/{prior['n_val']} | {fmt(prior['best_val_nll'])} | {per_modality} | "
                         f"{fmt(prior['gauss_val_nll'])} | {fmt(prior['empirical_val_nll'])} | {fmt(recon)} |")
    return "\n".join(lines) + "\n"


def write_csv(runs, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "condition", "modality", "sigma_scale", "task_id", "task_language", "episodes",
                         "success_rate", "bowl_choice_accuracy"])
        for run in runs:
            for task, row in sorted(run["tasks"].items(), key=lambda item: int(item[0])):
                writer.writerow([run["run"], run["condition"], run["modality"], run["sigma_scale"], task,
                                 row.get("task_language", ""), row.get("episodes"), row.get("success_rate"),
                                 row.get("bowl_choice_accuracy")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/vla"))
    parser.add_argument("--json", type=Path, help="Default: <out-dir>/results.json")
    parser.add_argument("--csv", type=Path, help="Default: <out-dir>/results_per_task.csv")
    parser.add_argument("--markdown", type=Path, help="Default: <out-dir>/results.md")
    args = parser.parse_args(argv)
    runs = load_runs(args.out_dir)
    priors = load_priors(args.out_dir)
    table = markdown(runs, priors)
    print(table, end="")
    (args.json or args.out_dir / "results.json").write_text(json.dumps({"runs": runs, "priors": priors}, indent=2))
    write_csv(runs, args.csv or args.out_dir / "results_per_task.csv")
    (args.markdown or args.out_dir / "results.md").write_text(table)


if __name__ == "__main__":
    main()
