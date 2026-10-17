"""Install the place_a2b_side task into a RoboTwin checkout (idempotent).

Copies leap/robotwin/tasks/place_a2b_side.py to <robotwin>/envs/, place_a2b_side.json to
<robotwin>/description/task_instruction/ and appends ``place_a2b_side: 400`` to
<robotwin>/task_config/_eval_step_limit.yml when absent.  ``--check`` only reports whether the
installation is complete and up to date (exit status 1 otherwise).
"""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "leap/robotwin/tasks"
TASK_NAME = "place_a2b_side"
STEP_LIMIT = 400


def planned_files(robotwin):
    """(source, destination) pairs copied by the installer."""
    robotwin = Path(robotwin)
    return [(TASK_DIR / f"{TASK_NAME}.py", robotwin / "envs" / f"{TASK_NAME}.py"),
            (TASK_DIR / f"{TASK_NAME}.json", robotwin / "description/task_instruction" / f"{TASK_NAME}.json")]


def step_limit_path(robotwin):
    return Path(robotwin) / "task_config/_eval_step_limit.yml"


def step_limit_installed(robotwin):
    path = step_limit_path(robotwin)
    if not path.exists():
        return False
    return any(line.split("#")[0].strip() == f"{TASK_NAME}: {STEP_LIMIT}" for line in path.read_text().splitlines())


def missing_items(robotwin):
    """Human-readable list of what is not installed or differs from the shipped files."""
    missing = []
    for source, destination in planned_files(robotwin):
        if not destination.exists():
            missing.append(f"{destination} is missing")
        elif destination.read_bytes() != source.read_bytes():
            missing.append(f"{destination} differs from {source}")
    if not step_limit_installed(robotwin):
        missing.append(f"{step_limit_path(robotwin)} has no '{TASK_NAME}: {STEP_LIMIT}' entry")
    return missing


def install(robotwin):
    """Copy the task files and add the step limit; returns the list of actions taken."""
    actions = []
    for source, destination in planned_files(robotwin):
        if not source.exists():
            raise FileNotFoundError(f"Shipped task file missing: {source}")
        if destination.exists() and destination.read_bytes() == source.read_bytes():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        actions.append(f"copied {source.relative_to(ROOT)} -> {destination}")
    path = step_limit_path(robotwin)
    if not step_limit_installed(robotwin):
        text = path.read_text() if path.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + f"{TASK_NAME}: {STEP_LIMIT}\n")
        actions.append(f"appended '{TASK_NAME}: {STEP_LIMIT}' to {path}")
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robotwin", type=Path, default=ROOT / "third_party/RoboTwin")
    parser.add_argument("--check", action="store_true", help="Report the state only; exit 1 if not installed")
    args = parser.parse_args()
    if not (args.robotwin / "envs").is_dir():
        print(f"Not a RoboTwin checkout: {args.robotwin}")
        sys.exit(2)
    if args.check:
        missing = missing_items(args.robotwin)
        for item in missing:
            print(item)
        print(f"{TASK_NAME}: {'installed' if not missing else 'NOT installed'} in {args.robotwin}")
        sys.exit(1 if missing else 0)
    actions = install(args.robotwin)
    for action in actions:
        print(action)
    if not actions:
        print(f"{TASK_NAME}: already installed in {args.robotwin}, nothing to do")


if __name__ == "__main__":
    main()
