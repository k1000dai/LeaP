"""Collect official RoboTwin eval results written by scripts/run_campaign.sh and intervention_sweep.sh.

Prints a Markdown table and writes JSON. Reads ROBOTWIN_ROOT/eval_result/<task>/leap.robotwin.deploy/
<task_config>/<ckpt_setting>/<timestamp>/_result.txt (official layout, newest timestamp per setting).
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--task-config", default="leap_clean")
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
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
