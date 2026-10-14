"""Opt-in interpreter shim: disable SAPIEN's ray-tracing denoiser.

RoboTwin's envs/_base_task.py asks SAPIEN for the "oidn" denoiser. On hosts
where OIDN has no usable CUDA backend (observed on an RTX 5080 / sm_120) SAPIEN
logs "OIDN Error: unsupported device type: CUDA" on every frame and the
collector intermittently hangs mid-episode. Rendering was already effectively
undenoised, so forcing "none" changes nothing in the observations.

Enabled only when LEAP_DISABLE_OIDN=1 and this directory is on PYTHONPATH.
Official RoboTwin files remain untouched.
"""
import os

if os.environ.get("LEAP_DISABLE_OIDN") == "1":
    try:
        import sapien.render as _render
    except Exception:  # SAPIEN absent or broken: do nothing
        _render = None
    if _render is not None:
        _original = _render.set_ray_tracing_denoiser

        def _set_ray_tracing_denoiser(name="none"):
            if name != "none":
                print(f"[leap shim] ray-tracing denoiser {name!r} -> 'none' (LEAP_DISABLE_OIDN=1)", flush=True)
            return _original("none")

        _render.set_ray_tracing_denoiser = _set_ray_tracing_denoiser


# Stall diagnostics: SIGUSR1 dumps every thread's Python stack to stderr.
if os.environ.get("LEAP_FAULTHANDLER") == "1":
    import faulthandler
    import signal
    import sys

    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)


# Vulkan/CUDA interop workaround: serve get_picture_cuda() from the CPU readback.
# RoboTwin's camera.py mixes camera.get_picture() (host readback) with
# camera.get_picture_cuda().torch() (Vulkan->CUDA external memory) on the same
# frames. On this host (RTX 5080, driver 580, SAPIEN 3.0.0b1) the host readback
# intermittently blocks forever inside get_picture("Color") during ray-traced
# collection. Both paths return bit-identical float32 images (verified), so
# with LEAP_NO_CUDA_INTEROP=1 the CUDA path is replaced by the host readback
# uploaded to the GPU with torch. Slightly slower; observations unchanged.
if os.environ.get("LEAP_NO_CUDA_INTEROP") == "1":
    try:
        import numpy as _np
        import torch as _torch
        from sapien.pysapien.render import RenderCameraComponent as _Camera
    except Exception:
        _Camera = None
    if _Camera is not None:
        class _HostPicture:
            """Minimal stand-in for sapien's CudaArray: only .torch() is used by RoboTwin."""

            def __init__(self, array):
                self._array = _np.ascontiguousarray(array)
                self.shape = self._array.shape

            def torch(self):
                return _torch.from_numpy(self._array).to("cuda" if _torch.cuda.is_available() else "cpu")

        _Camera.get_picture_cuda = lambda self, name: _HostPicture(self.get_picture(name))


# Renderer workaround: use SAPIEN's rasterizer instead of the ray tracer.
# With LEAP_RASTER=1, set_camera_shader_dir("rt") becomes "default" and the
# ray-tracing knobs become no-ops. Position/depth (and therefore the XYZ point
# clouds LeaP consumes) are geometric in both renderers; only RGB shading differs.
if os.environ.get("LEAP_RASTER") == "1":
    try:
        import sapien.render as _render2
    except Exception:
        _render2 = None
    if _render2 is not None:
        _orig_shader = _render2.set_camera_shader_dir

        def _set_camera_shader_dir(name="default"):
            if name == "rt":
                print("[leap shim] camera shader 'rt' -> 'default' (LEAP_RASTER=1)", flush=True)
                name = "default"
            return _orig_shader(name)

        _render2.set_camera_shader_dir = _set_camera_shader_dir
        for _fn in ("set_ray_tracing_samples_per_pixel", "set_ray_tracing_path_depth", "set_ray_tracing_denoiser"):
            setattr(_render2, _fn, lambda *a, **k: None)
