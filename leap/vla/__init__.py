"""Steering a frozen VLA (LeRobot pi0.5) through a learned flow-matching prior.

Modules (see docs/superpowers/specs/2026-10-17-vla-prior-steering-design.md):

- ``pi05_flow``: load the pi0.5 checkpoint, build its input batch like ``lerobot-eval``,
  cache the prefix (images + language) once per observation, run the flow forward
  (noise -> actions) or backwards (actions -> noise), and (un)normalise actions.
- ``prior``: the prior network predicting the source noise from VLA features and a
  steering embedding.
- ``libero_scene`` / ``instructions``: LIBERO env helpers and steering signals.

Importing this package stays light: LeRobot is imported lazily inside the functions
that need it, so the integrator math is testable without it.
"""
