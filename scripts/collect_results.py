"""Collect official RoboTwin eval results written by scripts/run_campaign.sh, intervention_sweep.sh and
run_steer_campaign.sh.

Prints a Markdown table and writes JSON. Reads ROBOTWIN_ROOT/eval_result/<task>/leap.robotwin.deploy/
<task_config>/<ckpt_setting>/<timestamp>/_result.txt (official layout, newest timestamp per setting).
With --side-logs DIR, the side-following diagnostics that place_a2b_side appends to $LEAP_STEER_LOG
(DIR/side_<variant>_<modality>.jsonl, one JSON line per check_success call with "side_correct" and
"success") are summarised per setting <variant>-s<seed>-<modality> in a second table.
"""
import argparse
import json
from pathlib import Path
import re

SETTING_PATTERN = re.compile(r"^(?P<variant>.+)-s(?P<seed>\d+)-(?P<modality>[a-z]+)$")


def side_log_name(setting):
    """logs/steer file stem for a run_steer_campaign.sh ckpt_setting, or None for other settings."""
    match = SETTING_PATTERN.match(setting)
    return f"side_{match['variant']}_{match['modality']}.jsonl" if match else None


def summarize_side_log(path):
    """Rates over the policy-rollout lines of one side log (lines marked as expert checks are ignored)."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lines = [line for line in lines if line.get("phase", "policy") == "policy"]
    if not lines:
        return {"episodes": 0}
    mean = lambda key: sum(bool(line.get(key)) for line in lines) / len(lines)  # noqa: E731
    return {"episodes": len(lines), "side_following": mean("side_correct"), "success": mean("success")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--task-config", default="leap_clean")
    parser.add_argument("--side-logs", type=Path, help="Directory holding side_<variant>_<modality>.jsonl files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = {}
    for result in sorted((args.robotwin / "eval_result").glob(f"*/leap.robotwin.deploy/{args.task_config}/*/*/_result.txt")):
        task, setting, stamp = result.parts[-6], result.parts[-3], result.parts[-2]
        rate = float(result.read_text().strip().splitlines()[-1])
        current = rows.setdefault(task, {}).get(setting)
        if current is None or stamp > current["timestamp"]:
            rows[task][setting] = {"success_rate": rate, "timestamp": stamp, "path": str(result)}
    settings = sorted({s for r in rows.values() for s in r})
    print("| task | " + " | ".join(settings) + " |")
    print("|---|" + "---|" * len(settings))
    for task in sorted(rows):
        print(f"| {task} | " + " | ".join(
            f"{100 * rows[task][s]['success_rate']:.0f}" if s in rows[task] else "" for s in settings) + " |")
    if args.side_logs:
        print("\n| task | setting | episodes in side log | side-following % | success % (log) | success % (official) |")
        print("|---|---|---|---|---|---|")
        for task in sorted(rows):
            for setting in sorted(rows[task]):
                name = side_log_name(setting)
                if name is None or not (args.side_logs / name).exists():
                    continue
                side = summarize_side_log(args.side_logs / name)
                rows[task][setting]["side_log"] = dict(side, path=str(args.side_logs / name))
                if side["episodes"]:
                    print(f"| {task} | {setting} | {side['episodes']} | {100 * side['side_following']:.0f} | "
                          f"{100 * side['success']:.0f} | {100 * rows[task][setting]['success_rate']:.0f} |")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
