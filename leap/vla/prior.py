"""Forward model of a frozen VLA's flow-matching prior (LeaP's mu/sigma head on top of pi0.5).

Flow Reversal Steering obtains, for a reference action chunk, the source noise that the
frozen VLA would denoise into it. ``VLAPrior`` predicts that noise *forward*, from the
VLA's own prefix features (mean-pooled image tokens of both cameras + the 8-d state, 4104-d)
and an optional steering embedding (768-d text / sketch / audio from the shared embedder):

    (mu, log sigma^2) = VLAPrior(feat, instr)        # each [B, horizon, action_dim]
    x0 = mu + sigma . eps,  eps ~ N(0, I)            # sample(); sigma_scale 0 -> mu
    x0_padded = pad_noise(x0)                        # dims >= action_dim ~ N(0, I), see pi05_flow

Inputs are standardised with per-dimension statistics of the training set, stored as
buffers so that a saved checkpoint carries them (``set_stats``).  The loss is the Gaussian
NLL of the reversal noise labels, masked by the chunk padding mask, optionally plus an MSE
term.  The head has the same (mu, log_var) output as ``leap.policy.LeaPPrior`` so later
variants can reuse LeaP's sampling and intervention code.  No LeRobot imports here: the
module is plain torch so the prior can be trained and tested without the VLA.
"""
import math

import numpy as np
import torch
import torch.nn as nn

LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 5.0
INSTR_MODES = ("none", "concat")
STD_FLOOR = 1e-6
LOG_2PI = math.log(2.0 * math.pi)


def _as_float_tensor(value, size, name):
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32).reshape(-1)
    if tensor.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {tuple(tensor.shape)}")
    return tensor


def valid_mask(pad_mask, mu):
    """Float mask [B, H, A] of the valid chunk elements (1 where the chunk was not padded)."""
    if pad_mask is None:
        return torch.ones_like(mu)
    keep = (~pad_mask.bool()).to(mu.dtype)
    if keep.ndim == mu.ndim - 1:
        keep = keep.unsqueeze(-1)
    return keep.expand_as(mu)


def gaussian_nll(mu, log_var, noise, pad_mask=None):
    """Per-element Gaussian NLL of ``noise`` under N(mu, exp(log_var)), averaged over the valid elements.

    ``pad_mask`` is [B, H] (True = padded chunk step) as stored by the reversal dataset; None
    counts every element.
    """
    keep = valid_mask(pad_mask, mu)
    nll = 0.5 * (log_var + (noise - mu) ** 2 * torch.exp(-log_var) + LOG_2PI)
    return (nll * keep).sum() / keep.sum().clamp_min(1.0)


def masked_mse(mu, noise, pad_mask=None):
    """Mean squared error between mu and the noise labels over the valid chunk elements."""
    keep = valid_mask(pad_mask, mu)
    return (((noise - mu) ** 2) * keep).sum() / keep.sum().clamp_min(1.0)


def standard_normal_nll(noise, pad_mask=None):
    """NLL of the labels under the VLA's default source N(0, I): the reference for an uninformative prior."""
    zeros = torch.zeros_like(noise)
    return gaussian_nll(zeros, zeros, noise, pad_mask)


class VLAPrior(nn.Module):
    """Diagonal Gaussian over the VLA's source noise, conditioned on prefix features (+ steering embedding).

    ``instr_mode``: "none" (offline DSBC without steering, the control) or "concat" (the
    standardised steering embedding is concatenated to the standardised features).
    """

    def __init__(self, feat_dim=4104, instr_dim=768, hidden=(1024, 1024), horizon=50, action_dim=7,
                 instr_mode="concat", dropout=0.1):
        super().__init__()
        if instr_mode not in INSTR_MODES:
            raise ValueError(f"instr_mode must be one of {INSTR_MODES}, got {instr_mode!r}")
        hidden = tuple(int(h) for h in hidden)
        if not hidden:
            raise ValueError("hidden must name at least one layer width")
        self.feat_dim = int(feat_dim)
        self.instr_dim = int(instr_dim)
        self.hidden = hidden
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.instr_mode = instr_mode
        self.dropout = float(dropout)
        self.register_buffer("feat_mean", torch.zeros(self.feat_dim))
        self.register_buffer("feat_std", torch.ones(self.feat_dim))
        self.register_buffer("instr_mean", torch.zeros(self.instr_dim))
        self.register_buffer("instr_std", torch.ones(self.instr_dim))
        in_dim = self.feat_dim + (self.instr_dim if instr_mode == "concat" else 0)
        layers = []
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * self.horizon * self.action_dim))
        self.net = nn.Sequential(*layers)

    # ---- configuration and statistics ----
    def config(self):
        return {"feat_dim": self.feat_dim, "instr_dim": self.instr_dim, "hidden": list(self.hidden),
                "horizon": self.horizon, "action_dim": self.action_dim, "instr_mode": self.instr_mode,
                "dropout": self.dropout}

    @torch.no_grad()
    def set_stats(self, feat_mean, feat_std, instr_mean=None, instr_std=None):
        """Store the training-set standardisation statistics (std floored at STD_FLOOR)."""
        self.feat_mean.copy_(_as_float_tensor(feat_mean, self.feat_dim, "feat_mean"))
        self.feat_std.copy_(_as_float_tensor(feat_std, self.feat_dim, "feat_std").clamp_min(STD_FLOOR))
        if (instr_mean is None) != (instr_std is None):
            raise ValueError("instr_mean and instr_std must be given together")
        if instr_mean is not None:
            self.instr_mean.copy_(_as_float_tensor(instr_mean, self.instr_dim, "instr_mean"))
            self.instr_std.copy_(_as_float_tensor(instr_std, self.instr_dim, "instr_std").clamp_min(STD_FLOOR))

    # ---- forward ----
    def forward(self, feat, instr=None):
        """feat [B, feat_dim], instr [B, instr_dim] (required iff instr_mode == "concat") -> mu, log_var [B, H, A]."""
        if feat.ndim != 2 or feat.shape[1] != self.feat_dim:
            raise ValueError(f"feat must be [B, {self.feat_dim}], got {tuple(feat.shape)}")
        x = (feat - self.feat_mean) / self.feat_std
        if self.instr_mode == "concat":
            if instr is None:
                raise ValueError("instr_mode 'concat' needs an instruction embedding")
            if instr.ndim != 2 or instr.shape[1] != self.instr_dim or instr.shape[0] != feat.shape[0]:
                raise ValueError(f"instr must be [B, {self.instr_dim}], got {tuple(instr.shape)}")
            x = torch.cat([x, (instr - self.instr_mean) / self.instr_std], dim=-1)
        elif instr is not None:
            raise ValueError("instr_mode 'none' takes no instruction embedding")
        out = self.net(x).reshape(feat.shape[0], 2, self.horizon, self.action_dim)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var

    # ---- loss and sampling ----
    def nll(self, mu, log_var, noise, pad_mask=None, mse_weight=0.0):
        """Masked Gaussian NLL (mean over valid elements) plus ``mse_weight`` times the masked MSE."""
        loss = gaussian_nll(mu, log_var, noise, pad_mask)
        if mse_weight:
            loss = loss + mse_weight * masked_mse(mu, noise, pad_mask)
        return loss

    @torch.no_grad()
    def sample(self, mu, log_var, sigma_scale=1.0, generator=None):
        """x0 = mu + sigma_scale . sigma . eps in the VLA's normalised action space ([B, H, A], 7-d, unpadded)."""
        if sigma_scale == 0:
            return mu.clone()
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return mu + sigma_scale * torch.exp(0.5 * log_var) * eps

    # ---- persistence ----
    def save(self, path, extra=None):
        """Plain torch.save of the config, the state dict (with the stat buffers) and an ``extra`` dict."""
        torch.save({"config": self.config(), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Returns (prior in eval mode, extra dict) from a file written by ``save``."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        prior = cls(**payload["config"])
        prior.load_state_dict(payload["state_dict"])
        prior.eval()
        return prior, payload.get("extra", {})
