"""Sketch instructions: a circle around the object and an arrow to the target on a camera frame.

Camera convention (verified on data/raw/beat_block_hammer/leap_clean/data/episode0.hdf5, frame of the
right-gripper grasp, see tests and the lead report): RoboTwin stores for every frame what SAPIEN's
``camera.get_intrinsic_matrix()`` / ``camera.get_extrinsic_matrix()`` return (envs/camera/camera.py
``get_config``), i.e.

- ``intrinsic_cv`` 3x3 OpenCV pinhole matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] in pixels;
- ``extrinsic_cv`` 3x4 world->camera matrix [R | t] in the OpenCV camera frame
  (x right, y down, z forward); a point is in front of the camera when z_cam > 0.

Pixel = (fx * x_cam / z_cam + cx, fy * y_cam / z_cam + cy) with u along the image width and v along
the height, origin at the top-left corner.  ``cam2world_gl`` (OpenGL model matrix) is not needed.
"""
import numpy as np
from PIL import Image, ImageDraw

CIRCLE_COLOR = (255, 0, 0)
ARROW_COLOR = (0, 80, 255)
CIRCLE_RADIUS_FRACTION = 22 / 320       # radius ~22 px on a 320-wide frame
CIRCLE_WIDTH_FRACTION = 3 / 320
ARROW_WIDTH_FRACTION = 5 / 320          # shaft ~5 px on a 320-wide frame
ARROW_HEAD_FRACTION = 16 / 320          # head length


def project_points(xyz, intrinsic_cv, extrinsic_cv):
    """World points [N, 3] -> pixel coordinates [N, 2] (u, v) under the convention above.

    Points at or behind the camera plane (z_cam <= 0) map to NaN.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(intrinsic_cv, dtype=np.float64).reshape(3, 3)
    E = np.asarray(extrinsic_cv, dtype=np.float64)
    if E.shape == (4, 4):
        E = E[:3]
    if E.shape != (3, 4):
        raise ValueError(f"extrinsic_cv must be 3x4 (or 4x4), got {E.shape}")
    cam = xyz @ E[:, :3].T + E[:, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    pixels = np.stack([u, v], axis=1)
    pixels[z <= 0] = np.nan
    return pixels


def clip_pixel(pixel, width, height):
    """Clamp a (u, v) pair into the image so markers at the border stay visible."""
    u, v = pixel
    if not np.isfinite(u) or not np.isfinite(v):
        raise ValueError("Point projects outside the camera frustum")
    return float(np.clip(u, 0, width - 1)), float(np.clip(v, 0, height - 1))


def sketch_pixels(intrinsic_cv, extrinsic_cv, object_xyz, target_xyz, width, height):
    """Clipped pixel coordinates of the object and the target: ((u, v), (u, v))."""
    pixels = project_points(np.stack([object_xyz, target_xyz]), intrinsic_cv, extrinsic_cv)
    return clip_pixel(pixels[0], width, height), clip_pixel(pixels[1], width, height)


def draw_sketch(image, object_px, target_px):
    """Draw the red circle and blue arrow in place on a PIL RGB image (sizes scale with the width)."""
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    radius = max(4, round(CIRCLE_RADIUS_FRACTION * width))
    circle_width = max(1, round(CIRCLE_WIDTH_FRACTION * width))
    shaft = max(1, round(ARROW_WIDTH_FRACTION * width))
    head = max(4, round(ARROW_HEAD_FRACTION * width))
    ox, oy = object_px
    tx, ty = target_px
    direction = np.array([tx - ox, ty - oy], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length > 1e-6:
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        # Start the shaft at the circle's rim and stop it under the head so the tip is at the target.
        start = np.array([ox, oy]) + unit * min(radius, length / 2)
        tip = np.array([tx, ty])
        base = tip - unit * head
        draw.line([tuple(start), tuple(base)], fill=ARROW_COLOR, width=shaft)
        half = head * 0.6
        draw.polygon([tuple(tip), tuple(base + normal * half), tuple(base - normal * half)], fill=ARROW_COLOR)
    draw.ellipse([ox - radius, oy - radius, ox + radius, oy + radius], outline=CIRCLE_COLOR, width=circle_width)
    return image


def render_sketch(rgb, intrinsic_cv, extrinsic_cv, object_xyz, target_xyz):
    """Return a PIL RGB image: ``rgb`` (HxWx3 uint8 or PIL) with the object circled and an arrow to the target."""
    image = rgb.convert("RGB").copy() if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    width, height = image.size
    object_px, target_px = sketch_pixels(intrinsic_cv, extrinsic_cv, object_xyz, target_xyz, width, height)
    return draw_sketch(image, object_px, target_px)
