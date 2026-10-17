"""Ensure benchmark launch delegates to the unchanged pinned official evaluator."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leap_launcher", ROOT / "scripts/robotwin.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
sys.modules.setdefault("robotwin", launcher)  # scripts/eval_limited.py imports its sibling by name
spec = importlib.util.spec_from_file_location("leap_limited_launcher", ROOT / "scripts/eval_limited.py")
limited = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limited)


class OfficialLauncherTest(unittest.TestCase):
    def test_official_eval_command(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "latest.ckpt"
            checkpoint.touch()
            with patch.object(launcher, "verified_root", return_value=Path(temp)), \
                 patch.object(launcher.subprocess, "run") as run, \
                 patch.object(sys, "argv", ["robotwin.py", "eval", "--robotwin", temp,
                                           "--task", "beat_block_hammer", "--gpu", "2",
                                           "--checkpoint", str(checkpoint), "--seed", "3",
                                           "--ckpt-setting", "noprior-s3",
                                           "--intervention", '{"sigma_scale": 0}',
                                           "--instruction-modality", "sketch", "--embedder", "org/model",
                                           "--instr-dump-dir", "dumps/run1"]):
                def inspect(cmd, **kwargs):
                    self.assertEqual(cmd[:3], [sys.executable, "script/eval_policy.py", "--config"])
                    config = yaml.safe_load(Path(cmd[3]).read_text())
                    self.assertEqual(config["policy_name"], "leap.robotwin.deploy")
                    self.assertEqual(config["seed"], 3)
                    self.assertEqual(config["ckpt_setting"], "noprior-s3")
                    self.assertEqual(config["intervention"], {"sigma_scale": 0})
                    self.assertEqual(config["instruction_modality"], "sketch")
                    self.assertEqual(config["embedder"], "org/model")
                    self.assertIsNone(config["tts_voice"])
                    self.assertEqual(config["instr_dump_dir"], str(Path("dumps/run1").resolve()))
                    self.assertNotIn("test_num_override", config)
                    self.assertNotIn("seed_list", config)
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
                    self.assertEqual(kwargs["cwd"], Path(temp))
                run.side_effect = inspect
                launcher.main()
                run.assert_called_once()

    def test_default_eval_has_no_intervention_and_rejects_bad_setting(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "latest.ckpt"
            checkpoint.touch()
            base = ["robotwin.py", "eval", "--robotwin", temp, "--task", "t", "--gpu", "0",
                    "--checkpoint", str(checkpoint)]
            with patch.object(launcher, "verified_root", return_value=Path(temp)), \
                 patch.object(launcher.subprocess, "run") as run:
                def inspect(cmd, **kwargs):
                    config = yaml.safe_load(Path(cmd[3]).read_text())
                    self.assertEqual(config["ckpt_setting"], "leap")
                    self.assertIsNone(config["intervention"])
                    self.assertEqual(config["instruction_modality"], "none")
                    self.assertEqual(config["embedder"], "google/embeddinggemma-2")
                    self.assertIsNone(config["tts_voice"])
                    self.assertIsNone(config["instr_dump_dir"])
                run.side_effect = inspect
                with patch.object(sys, "argv", base):
                    launcher.main()
                with patch.object(sys, "argv", base + ["--ckpt-setting", "../x"]), \
                     self.assertRaises(SystemExit):
                    launcher.main()

    def test_audio_modality_requires_existing_tts_voice(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "latest.ckpt"
            checkpoint.touch()
            voice = Path(temp) / "en_US-lessac-medium.onnx"
            base = ["robotwin.py", "eval", "--robotwin", temp, "--task", "t", "--gpu", "0",
                    "--checkpoint", str(checkpoint), "--instruction-modality", "audio"]
            with patch.object(launcher, "verified_root", return_value=Path(temp)), \
                 patch.object(launcher.subprocess, "run") as run:
                for argv in (base, base + ["--tts-voice", str(voice)]):  # missing flag, missing file
                    with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                        launcher.main()
                run.assert_not_called()
                voice.touch()

                def inspect(cmd, **kwargs):
                    config = yaml.safe_load(Path(cmd[3]).read_text())
                    self.assertEqual(config["instruction_modality"], "audio")
                    self.assertEqual(config["tts_voice"], str(voice.resolve()))
                run.side_effect = inspect
                with patch.object(sys, "argv", base + ["--tts-voice", str(voice)]):
                    launcher.main()
                run.assert_called_once()
            with patch.object(sys, "argv", base + ["--instruction-modality", "speech"]), \
                 self.assertRaises(SystemExit):
                launcher.main()

    def test_limited_launcher_shares_instruction_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "latest.ckpt"
            checkpoint.touch()
            base = ["eval_limited.py", "--robotwin", temp, "--task", "t", "--gpu", "0",
                    "--checkpoint", str(checkpoint), "--episodes", "2"]
            with patch.object(limited, "verified_root", side_effect=AssertionError("validated too late")):
                with patch.object(sys, "argv", base + ["--instruction-modality", "audio"]), \
                     self.assertRaises(SystemExit):
                    limited.main()
            class StopAfterChecks(Exception):
                pass

            with patch.object(limited, "verified_root", side_effect=StopAfterChecks) as root, \
                 patch.object(sys, "argv", base + ["--instruction-modality", "text", "--embedder", "m"]), \
                 self.assertRaises(StopAfterChecks):  # the flags were accepted and validated
                limited.main()
            root.assert_called_once()

    def test_modified_official_entry_rejected(self):
        import json
        commit = json.loads((ROOT / "third_party/robotwin.lock.json").read_text())["commit"]
        with tempfile.TemporaryDirectory() as temp:
            entry = Path(temp) / "script/eval_policy.py"
            entry.parent.mkdir()
            entry.write_text("modified")
            with patch.object(launcher.subprocess, "check_output", side_effect=[commit + "\n", b"official"]):
                with self.assertRaisesRegex(RuntimeError, "differs from official"):
                    launcher.verified_root(Path(temp), "script/eval_policy.py")


if __name__ == "__main__":
    unittest.main()
