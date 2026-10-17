"""Dry-run checks for scripts/run_steer_campaign.sh and the side-log summary in scripts/collect_results.py."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_steer_campaign.sh"
spec = importlib.util.spec_from_file_location("collect_results", ROOT / "scripts/collect_results.py")
collect_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_results)


def dry_run(temp, **env):
    """Run the driver with DRY_RUN=1 against an empty RoboTwin root; everything it writes goes under temp."""
    robotwin = Path(temp) / "robotwin"
    (robotwin / "task_config").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, DRY_RUN="1", ROBOTWIN_ROOT=str(robotwin), LOG_DIR=str(Path(temp) / "logs"),
                       OUTPUTS=str(Path(temp) / "outputs"), **env)
    result = subprocess.run(["bash", str(DRIVER)], cwd=ROOT, env=environment, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


class SteerCampaignDriverTest(unittest.TestCase):
    def test_syntax(self):
        subprocess.run(["bash", "-n", str(DRIVER)], check=True)

    def test_dry_run_prints_every_stage_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            code, out = dry_run(temp)
        self.assertEqual(code, 0, out)
        commands = [line.split("+ ", 1)[1] for line in out.splitlines() if "] + " in line]
        self.assertIn("python scripts/install_task.py", commands[0])
        self.assertIn("--check", commands[1])
        self.assertIn("scripts/robotwin.py prepare", commands[2])
        self.assertIn("--episodes 100", commands[2])
        self.assertIn("scripts/robotwin.py collect", commands[3])
        self.assertIn("scripts/robotwin.py convert --input data/raw/place_a2b_side/leap_steer/data", commands[4])
        self.assertIn("scripts/make_instructions.py --raw data/raw/place_a2b_side/leap_steer", commands[5])
        self.assertIn("scripts/embed_instructions.py --bundle", commands[6])
        self.assertIn("scripts/instr_probe.py", commands[7])
        trains = [c for c in commands if "leap.robotwin.train" in c]
        self.assertEqual([c.split("--config configs/steer_")[1].split(".yaml")[0] for c in trains],
                         ["noinstr", "prior", "cond", "both"])
        self.assertNotIn("--instr", trains[0])
        for command in trains[1:]:
            self.assertIn("--instr data/place_a2b_side-leap_steer-100.instr", command)
        self.assertEqual({c.split("[gpu ")[1][0] for c in trains}, {"0", "1", "2"})
        evals = [c for c in commands if "scripts/robotwin.py eval" in c]
        settings = [c.split("--ckpt-setting ")[1].split()[0] for c in evals]
        self.assertEqual(settings, ["noinstr-s0-none"] + [f"{v}-s0-{m}" for v in ("prior", "cond", "both")
                                                           for m in ("text", "sketch", "audio")])
        self.assertIn("--instruction-modality none", evals[0])
        self.assertNotIn("--tts-voice", evals[0])
        for command, setting in zip(evals[1:], settings[1:]):
            self.assertIn(f"--instruction-modality {setting.rsplit('-', 1)[1]}", command)
            self.assertIn("--tts-voice /", command)
            self.assertIn("--embedder google/embeddinggemma-2", command)
            self.assertIn(f"LEAP_STEER_LOG={temp}/logs/side_{setting.split('-')[0]}_{setting.rsplit('-', 1)[1]}.jsonl",
                          command)
            self.assertIn(f"--gpu {command.split('[gpu ')[1][0]}", command)
        self.assertIn("scripts/collect_results.py", commands[-1])
        self.assertIn("--side-logs", commands[-1])
        self.assertFalse((ROOT / "outputs/place_a2b_side-noinstr-s0").exists())

    def test_stage_selection_and_skips(self):
        with tempfile.TemporaryDirectory() as temp:
            # A finished training run is recognised by its DONE marker; a finished evaluation by _result.txt.
            done = Path(temp) / "outputs/place_a2b_side-prior-s0"
            (done / "checkpoints").mkdir(parents=True)
            (done / "checkpoints/latest.ckpt").touch()
            (done / "DONE").touch()
            result = Path(temp) / "robotwin/eval_result/place_a2b_side/leap.robotwin.deploy/leap_steer/cond-s0-text/2026"
            result.mkdir(parents=True)
            (result / "_result.txt").write_text("Instruction Type: unseen\n\n0.42\n")
            code, out = dry_run(temp, STAGES="train eval", TRAIN_PARALLEL="4")
        self.assertEqual(code, 0, out)
        self.assertNotIn("install_task", out)
        self.assertNotIn("robotwin.py collect", out)
        self.assertNotIn("steer_prior.yaml", out)
        self.assertIn("training complete", out)
        self.assertEqual(out.count("leap.robotwin.train"), 3)
        self.assertIn("done: success_rate=0.42", out)
        self.assertEqual(out.count("robotwin.py eval"), 9)

    def test_refuses_to_run_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "logs").mkdir()
            (Path(temp) / "logs/campaign.pid").write_text(str(os.getpid()))
            code, out = dry_run(temp)
        self.assertEqual(code, 2)
        self.assertIn("another campaign is running", out)


class SideLogSummaryTest(unittest.TestCase):
    def test_side_logs_join_official_results(self):
        with tempfile.TemporaryDirectory() as temp:
            robotwin, logs = Path(temp) / "robotwin", Path(temp) / "logs"
            logs.mkdir()
            for setting, rate in (("prior-s0-text", 0.8), ("noinstr-s0-none", 0.3), ("leap-s0", 0.9)):
                result = robotwin / f"eval_result/place_a2b_side/leap.robotwin.deploy/leap_steer/{setting}/2026-01"
                result.mkdir(parents=True)
                (result / "_result.txt").write_text(f"Instruction Type: unseen\n\n{rate}\n")
            # One line per evaluated episode (ep_num); a repeated ep_num counts once.
            lines = [{"side": "left", "side_correct": True, "success": True, "ep_num": i} for i in range(3)]
            lines += [{"side": "right", "side_correct": False, "success": False, "ep_num": 3}]
            lines += [{"side": "right", "side_correct": False, "success": False, "ep_num": 3}]
            (logs / "side_prior_text.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            output = Path(temp) / "results.json"
            with patch.object(sys, "argv", ["collect_results.py", "--robotwin", str(robotwin), "--task-config",
                                            "leap_steer", "--side-logs", str(logs), "--output", str(output)]):
                collect_results.main()
            rows = json.loads(output.read_text())["place_a2b_side"]
        self.assertEqual(rows["prior-s0-text"]["side_log"]["episodes"], 4)
        self.assertAlmostEqual(rows["prior-s0-text"]["side_log"]["side_following"], 0.75)
        self.assertAlmostEqual(rows["prior-s0-text"]["side_log"]["success"], 0.75)
        self.assertNotIn("side_log", rows["noinstr-s0-none"])
        self.assertNotIn("side_log", rows["leap-s0"])
        self.assertIsNone(collect_results.side_log_name("leap-s0-sigma0"))
        self.assertEqual(collect_results.side_log_name("both-s1-audio"), "side_both_audio.jsonl")


if __name__ == "__main__":
    unittest.main()
