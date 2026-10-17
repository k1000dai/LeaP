"""Limited-episode evaluation through the official evaluator (NOT a benchmark score).

Like scripts/smoke_robotwin.py, this calls the unchanged official eval_policy()
but with test_num=--episodes instead of 100, and exits before main() writes a
score with its hard-coded denominator. Intended for degenerate prior
interventions whose outcome is unambiguous after a few episodes, where the
official 100-episode run would waste hours on 400-step failures. Results must
be reported as "<successes>/<episodes>, limited protocol".
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from robotwin import ROOT, add_instruction_arguments, instruction_config, verified_root


class LimitedComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="leap_clean")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--intervention", type=json.loads, default=None)
    parser.add_argument("--output", type=Path, help="JSON result file")
    add_instruction_arguments(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes >= 100:
        parser.error("--episodes must be in [1, 99]; use scripts/robotwin.py eval for the official 100")
    instructions = instruction_config(args, parser)
    root = verified_root(args.robotwin, "script/eval_policy.py")
    checkpoint = args.checkpoint.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(root)
    sys.path[:0] = [str(ROOT), str(root), str(root / "script")]
    spec = importlib.util.spec_from_file_location("official_eval", root / "script/eval_policy.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from test_render import Sapien_TEST
    Sapien_TEST()
    evaluate = official.eval_policy

    def limited(*positional, **kwargs):
        kwargs["test_num"] = args.episodes
        next_seed, successes = evaluate(*positional, **kwargs)
        result = {"protocol": "limited", "benchmark_score": False, "task": args.task,
                  "episodes": args.episodes, "successes": int(successes),
                  "success_rate": successes / args.episodes, "intervention": args.intervention,
                  "instruction_modality": args.instruction_modality,
                  "checkpoint": str(checkpoint), "seed": args.seed, "next_seed": next_seed}
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
        raise LimitedComplete

    official.eval_policy = limited
    try:
        official.main({"policy_name": "leap.robotwin.deploy", "task_name": args.task,
                       "task_config": args.task_config, "ckpt_setting": "limited-only",
                       "ckpt_path": str(checkpoint), "seed": args.seed,
                       "instruction_type": "unseen", "device": "cuda:0",
                       "intervention": args.intervention, **instructions})
    except LimitedComplete:
        pass


if __name__ == "__main__":
    main()
