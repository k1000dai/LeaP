"""make_instructions.make_sketch must draw on a white canvas, exactly like the evaluator's render_sketch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_instructions.py"
# Head camera of data/raw/beat_block_hammer/leap_clean/data/episode0.hdf5 (320x240).
INTRINSIC = np.array([[277.13, 0.0, 160.0], [0.0, 277.13, 120.0], [0.0, 0.0, 1.0]])
EXTRINSIC = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.75], [0.0, 1.0, 0.0, 0.6]])


def load_script():
    spec = importlib.util.spec_from_file_location("make_instructions_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MakeSketchTest(unittest.TestCase):
    def test_sketch_is_white_canvas_with_drawing(self):
        module = load_script()
        photo = Image.fromarray(np.full((240, 320, 3), 120, dtype=np.uint8))
        steer = {"side": "right", "A_xyz": [-0.1, 0.3, 0.0], "target_xyz": [0.1, 0.3, 0.0]}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(module, "load_frame0", return_value=(photo, INTRINSIC, EXTRINSIC)):
            sketch_path, meta_path = Path(temp) / "sketch.png", Path(temp) / "sketch_meta.json"
            module.make_sketch(Path(temp) / "episode0.hdf5", steer, sketch_path, meta_path)
            pixels = np.asarray(Image.open(sketch_path).convert("RGB"))
            meta = json.loads(meta_path.read_text())
        self.assertEqual(pixels.shape, (240, 320, 3))
        white = (pixels == 255).all(axis=-1).mean()
        self.assertGreater(white, 0.9, "the sketch must not be drawn on the camera photo")
        self.assertGreater(((pixels[..., 0] == 255) & (pixels[..., 1] == 0)).sum(), 50)   # red circle
        self.assertGreater(((pixels[..., 2] == 255) & (pixels[..., 0] == 0)).sum(), 50)   # blue arrow
        self.assertEqual(meta["side"], "right")
        self.assertEqual(meta["image_size"], [320, 240])


if __name__ == "__main__":
    unittest.main()
