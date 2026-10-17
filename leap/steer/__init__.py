"""Multimodal instruction steering: sketch rendering, TTS, embedding and instruction bundles.

Importing this package stays light (numpy + PIL); the embedder and TTS import their
heavy dependencies only when an instance is first used.
"""
from .bundle import (AUDIO_VARIANTS, INSTR_DIM, MODALITIES, SIDE_INDEX, TEXT_VARIANTS, bundle_path,
                     check_bundle, load_embeddings, standardize_stats)
from .sketch import project_points, render_sketch

__all__ = ["AUDIO_VARIANTS", "INSTR_DIM", "MODALITIES", "SIDE_INDEX", "TEXT_VARIANTS", "bundle_path",
           "check_bundle", "load_embeddings", "standardize_stats", "project_points", "render_sketch"]
