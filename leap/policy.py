"""LeaP — Learnable source Prior for generative robot policies.

A generative visuomotor policy that replaces the uninformative N(0, I) source
of flow matching with a *learned*, state-conditioned diagonal Gaussian prior:

    (mu, log sigma^2) = LeaPPrior(state)              # per action dim
    x0 = mu + sigma . eps,   eps ~ N(0, I)            # reparameterized source
    x1 = FlowMatch(v_theta, start=x0, cond=z_obs)     # linear-OT flow to action

The prior mu/sigma are produced from the proprioceptive state only; the point
cloud is used to condition the flow velocity network (z_obs). Training combines
three terms:

    L = L_flow  +  w_nll . L_nll  +  w_align . L_align

  * L_flow : conditional flow-matching MSE on the linear OT path
  * L_nll  : Gaussian NLL of the ground-truth action chunk under (mu, sigma),
             which is what makes the prior *informative* (pulls x0 near x1)
  * L_align: symmetric InfoNCE between the source sample and the action chunk,
             a CLIP-style alignment that sharpens the state->action coupling

This single file collapses the original 4-level research inheritance into one
self-contained class; behavior is identical to the paper's main model.

Instruction steering (``instr_mode``, default "none" = the model above): a
768-d instruction embedding ``obs["instr"]`` (B, instr_dim), standardised
with stored training statistics and passed through an adapter MLP to z_e, can
enter the prior head ("prior"), the velocity-net condition ("cond") or both.
"""
from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from .base_policy import BasePolicy
from .normalizer import LinearNormalizer
from .pointnet import PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp
from .flow_net import SimpleFlowNet
from .flow_matcher import SimpleConditionalFlowMatcher
from .model_util import print_params


def _make_mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    if len(hidden_dims) == 0:
        raise RuntimeError("MLP hidden_dims is empty")
    net_arch = [] if len(hidden_dims) == 1 else list(hidden_dims[:-1])
    output_dim = hidden_dims[-1]
    layers = create_mlp(in_dim, output_dim, net_arch, nn.ReLU)
    if output_dim != out_dim:
        layers.append(nn.Linear(output_dim, out_dim))
    return nn.Sequential(*layers)


class LeaPPrior(nn.Module):
    """Learnable diagonal-Gaussian source prior conditioned on robot state.

    state (B, To, state_dim) -> state encoder -> MLP head -> (mu, log_var),
    each of size action_dim. A single (mu, sigma) is broadcast across the H
    prediction steps (shared-across-chunk prior).
    """

    def __init__(self, state_dim, action_dim, n_obs_steps,
                 encoder_output_dim=128, encoder_hidden_dims=(128, 128),
                 learnable=True, instr_feature_dim=0):
        super().__init__()
        if instr_feature_dim and not learnable:
            raise ValueError("A NoPrior source has no head to receive an instruction feature")
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.learnable = learnable
        self.instr_feature_dim = instr_feature_dim
        self.state_encoder = _make_mlp(state_dim, encoder_hidden_dims, encoder_output_dim)
        dist_feature_dim = encoder_output_dim * n_obs_steps
        # NoPrior baseline keeps the state encoder (flow condition) but has no
        # distribution head: its source is the standard N(0, I). With an
        # instruction feature the head sees cat(z_s over To, z_e).
        head_dim = dist_feature_dim + instr_feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim * 2),     # -> (mu, log_var)
        ) if learnable else None

    def forward(self, state, instr_feat=None):
        """state: (B, To, state_dim) -> mu, log_var each (B, action_dim), plus z_s.

        instr_feat (B, instr_feature_dim) is required iff the prior was built
        with instr_feature_dim > 0 and is concatenated to the head input.
        For the NoPrior baseline mu is zero and log_var is zero (sigma = 1),
        detached from the graph, so the same sampling code yields N(0, I).
        """
        B, To = state.shape[:2]
        z_s = self.state_encoder(state.reshape(-1, state.shape[-1])).reshape(B, To, -1)
        if self.head is None:
            zeros = torch.zeros((B, self.action_dim), device=state.device, dtype=state.dtype)
            return zeros, zeros.clone(), z_s
        head_input = z_s.reshape(B, -1)
        if (instr_feat is None) != (self.instr_feature_dim == 0):
            raise ValueError("instr_feat must be given exactly when the prior has instr_feature_dim > 0")
        if instr_feat is not None:
            head_input = torch.cat([head_input, instr_feat], dim=-1)
        dist_params = self.head(head_input)
        mu, log_var = dist_params.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var, z_s

    def sample(self, mu, log_var, horizon):
        """Reparameterized source x0 = mu + sigma . eps, broadcast over H."""
        eps = torch.randn((mu.shape[0], horizon, mu.shape[-1]), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + torch.exp(0.5 * log_var).unsqueeze(1) * eps


class LeaPPolicy(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 encoder_output_dim: int = 128,
                 encoder_hidden_dims=(128, 128),
                 use_pc_color: bool = False,
                 pointcloud_encoder_cfg=None,
                 # ---- losses ----
                 nll_loss_weight: float = 1.0,
                 align_loss_weight: float = 0.3,
                 align_temperature: float = 0.1,
                 # ---- flow matcher / net ----
                 num_sampling_steps: int = 6,
                 flow_matcher_sigma: float = 0.0,
                 flow_net_hidden_dim: int = 512,
                 flow_net_num_layers: int = 4,
                 flow_net_mlp_ratio: float = 4.0,
                 flow_net_dropout: float = 0.0,
                 flow_net_time_embed_dim: int = 256,
                 # ---- "leap" (learned source) or "none" (N(0, I) NoPrior baseline) ----
                 prior_type: str = "leap",
                 # ---- instruction entry point: "none" | "prior" | "cond" | "both" ----
                 instr_mode: str = "none",
                 instr_dim: int = 768,
                 instr_adapter_hidden: int = 256,
                 instr_feature_dim: int = 128,
                 **kwargs):
        super().__init__()
        if prior_type not in ("leap", "none"):
            raise ValueError(f"prior_type must be 'leap' or 'none', got {prior_type!r}")
        if instr_mode not in ("none", "prior", "cond", "both"):
            raise ValueError(f"instr_mode must be one of none/prior/cond/both, got {instr_mode!r}")
        instr_to_prior = instr_mode in ("prior", "both")
        instr_to_cond = instr_mode in ("cond", "both")
        if instr_to_prior and prior_type == "none":
            raise ValueError(f"instr_mode={instr_mode!r} needs a learnable prior head; prior_type is 'none'")

        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        obs_meta = shape_meta['obs']
        pc_in_dim = obs_meta['point_cloud']['shape'][-1]
        state_dim = obs_meta['agent_pos']['shape'][-1]
        if not use_pc_color:
            pc_in_dim = min(pc_in_dim, 3)
        if instr_mode != "none" and "instr" in obs_meta and obs_meta["instr"]["shape"][-1] != instr_dim:
            raise ValueError(f"shape_meta instr dim {obs_meta['instr']['shape'][-1]} != instr_dim {instr_dim}")

        # ---- point-cloud encoder (DP3 PointNet) ----
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = dict(
                in_channels=pc_in_dim, out_channels=encoder_output_dim,
                use_layernorm=True, final_norm="layernorm")
        else:
            pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg)
            pointcloud_encoder_cfg["in_channels"] = pc_in_dim
            pointcloud_encoder_cfg["out_channels"] = encoder_output_dim
        self.pc_encoder = (PointNetEncoderXYZRGB if use_pc_color else PointNetEncoderXYZ)(**pointcloud_encoder_cfg)

        # ---- instruction adapter: standardised embedding -> z_e (only with instr_mode) ----
        if instr_mode != "none":
            self.register_buffer("instr_mean", torch.zeros(instr_dim))
            self.register_buffer("instr_std", torch.ones(instr_dim))
            self.instr_adapter = nn.Sequential(
                nn.Linear(instr_dim, instr_adapter_hidden), nn.ReLU(),
                nn.Linear(instr_adapter_hidden, instr_feature_dim))
        else:
            self.instr_adapter = None

        # ---- LeaP learnable source prior (state-only, plus z_e for "prior"/"both") ----
        self.prior = LeaPPrior(state_dim, action_dim, n_obs_steps,
                               encoder_output_dim, encoder_hidden_dims,
                               learnable=prior_type == "leap",
                               instr_feature_dim=instr_feature_dim if instr_to_prior else 0)

        # ---- alignment projection heads (InfoNCE); unused by NoPrior ----
        if prior_type == "leap":
            self.align_action_proj = _make_mlp(horizon * action_dim, encoder_hidden_dims, encoder_output_dim)
            self.align_obs_proj = _make_mlp(horizon * action_dim, encoder_hidden_dims, encoder_output_dim)
        else:
            self.align_action_proj = self.align_obs_proj = None

        # ---- flow velocity net + matcher (operate on flattened H*A chunk) ----
        flow_dim = horizon * action_dim
        # obs condition = cat(z_pc, z_s) flattened over To, plus z_e for "cond"/"both"
        self.obs_feature_dim = (encoder_output_dim * 2) * n_obs_steps
        self.flow_net = SimpleFlowNet(
            input_dim=flow_dim, hidden_dim=flow_net_hidden_dim, output_dim=flow_dim,
            num_layers=flow_net_num_layers, mlp_ratio=flow_net_mlp_ratio,
            dropout=flow_net_dropout, time_embed_dim=flow_net_time_embed_dim,
            condition_dim=self.obs_feature_dim + (instr_feature_dim if instr_to_cond else 0))
        self.flow_matcher = SimpleConditionalFlowMatcher(
            num_sampling_steps=num_sampling_steps, sigma=flow_matcher_sigma)

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.encoder_output_dim = encoder_output_dim
        self.use_pc_color = use_pc_color
        self.num_sampling_steps = num_sampling_steps
        self.nll_loss_weight = nll_loss_weight
        self.align_loss_weight = align_loss_weight
        self.align_temperature = align_temperature
        self.prior_type = prior_type
        self.instr_mode = instr_mode
        self.instr_dim = instr_dim
        self.instr_feature_dim = instr_feature_dim
        self.prior_intervention = None

        if prior_type == "leap":
            cprint("[LeaP] learnable source prior (state -> mu/sigma) + linear-OT flow matcher", "green")
            cprint(f"[LeaP] w_nll={nll_loss_weight}  w_align={align_loss_weight}  tau={align_temperature}  "
                   f"NFE={num_sampling_steps}", "green")
        else:
            cprint(f"[LeaP] NoPrior baseline: N(0, I) source + linear-OT flow matcher  NFE={num_sampling_steps}",
                   "yellow")
        if instr_mode != "none":
            cprint(f"[LeaP] instruction {instr_dim}-d -> adapter -> {instr_feature_dim}-d entering: "
                   f"{'prior head' if instr_to_prior else ''}{' + ' if instr_mode == 'both' else ''}"
                   f"{'velocity-net condition' if instr_to_cond else ''}", "green")
        print_params(self)

    # ------------------------------------------------------------------
    # Instruction embedding statistics (set from the training bundle, stored
    # as buffers so checkpoints carry them)
    # ------------------------------------------------------------------
    def set_instr_stats(self, mean, std):
        """Per-dimension mean/std used to standardise obs["instr"] before the adapter."""
        if self.instr_mode == "none":
            raise RuntimeError("set_instr_stats is only meaningful when instr_mode != 'none'")
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.instr_dim or std.numel() != self.instr_dim:
            raise ValueError(f"instr stats must have {self.instr_dim} entries")
        if not torch.isfinite(mean).all() or not (std > 0).all():
            raise ValueError("instr stats must be finite with strictly positive std")
        with torch.no_grad():
            self.instr_mean.copy_(mean)
            self.instr_std.copy_(std)

    # ------------------------------------------------------------------
    # Inference-time prior intervention (default off; training unaffected)
    # ------------------------------------------------------------------
    def set_prior_intervention(self, intervention=None):
        """Override the learned source at inference. ``None`` restores the default.

        Keys (all optional):
          sigma_scale     float   multiply sigma (0 -> deterministic source at mu)
          mu_shift        [A]     add to mu (normalized action units)
          mu_override     [A]     replace mu
          logvar_override [A]     replace log sigma^2
          direct_mu       bool    skip the flow; return mu broadcast over the horizon
        """
        if intervention is None:
            self.prior_intervention = None
            return
        if not isinstance(intervention, dict):
            raise TypeError(f"Prior intervention must be a dict or None, got {type(intervention).__name__}")
        allowed = {"sigma_scale", "mu_shift", "mu_override", "logvar_override", "direct_mu"}
        unknown = set(intervention) - allowed
        if unknown:
            raise ValueError(f"Unknown prior intervention keys: {sorted(unknown)}")
        if "sigma_scale" in intervention and float(intervention["sigma_scale"]) < 0:
            raise ValueError("sigma_scale must be non-negative")
        self.prior_intervention = dict(intervention)

    def _apply_prior_intervention(self, mu, log_var):
        cfg = self.prior_intervention
        if not cfg:
            return mu, log_var

        def as_vec(value):
            return torch.as_tensor(value, device=mu.device, dtype=mu.dtype).reshape(1, -1).expand_as(mu)

        if "mu_override" in cfg:
            mu = as_vec(cfg["mu_override"])
        if "mu_shift" in cfg:
            mu = mu + as_vec(cfg["mu_shift"])
        if "logvar_override" in cfg:
            log_var = as_vec(cfg["logvar_override"])
        if "sigma_scale" in cfg:
            scale = float(cfg["sigma_scale"])
            if scale == 0:
                log_var = torch.full_like(log_var, -1e4)   # sigma == 0 exactly in fp32
            else:
                log_var = log_var + 2.0 * math.log(scale)
        return mu, log_var

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    @staticmethod
    def _split_instr(obs_dict):
        """obs_dict -> (normalisable obs, raw instr or None).

        "instr" is a raw embedding that the LinearNormalizer does not know; it
        is standardised by the policy's own instr_mean/instr_std instead.
        """
        obs = {key: value for key, value in obs_dict.items() if key != "instr"}
        return obs, obs_dict.get("instr")

    def _encode_instr(self, instr):
        """instr (B, instr_dim) float -> z_e (B, instr_feature_dim)."""
        if instr is None:
            raise ValueError(f"instr_mode={self.instr_mode!r} requires obs['instr'] of shape (B, {self.instr_dim})")
        instr = torch.as_tensor(instr).to(self.instr_mean)
        if instr.ndim != 2 or instr.shape[-1] != self.instr_dim:
            raise ValueError(f"obs['instr'] must have shape (B, {self.instr_dim}), got {tuple(instr.shape)}")
        return self.instr_adapter((instr - self.instr_mean) / self.instr_std)

    def _encode_obs(self, nobs, instr=None):
        """Returns (mu, log_var, z_obs) where z_obs is the flow condition.

        instr is the raw instruction embedding (B, instr_dim); it is ignored
        when instr_mode is "none" and required otherwise.
        """
        pc = nobs["point_cloud"]
        if not self.use_pc_color:
            pc = pc[..., :3]
        state = nobs["agent_pos"]
        pc = pc[:, :self.n_obs_steps]
        state = state[:, :self.n_obs_steps]
        B, To, npts, cdim = pc.shape

        pc_feat = self.pc_encoder(pc.reshape(B * To, npts, cdim)).reshape(B, To, -1)
        z_e = self._encode_instr(instr) if self.instr_mode != "none" else None
        # prior from state only, or state + instruction for "prior"/"both"
        mu, log_var, z_s = self.prior(state, z_e if self.prior.instr_feature_dim else None)
        z_obs = torch.cat([pc_feat, z_s], dim=-1).reshape(B, -1)   # flow condition uses pc + state
        if self.instr_mode in ("cond", "both"):
            z_obs = torch.cat([z_obs, z_e], dim=-1)                # ... + instruction
        return mu, log_var, z_obs

    def _flow_velocity(self, x_flat, t, global_cond):
        return self.flow_net(x_flat, t, global_cond=global_cond)

    # ---- losses ----
    def _gaussian_nll(self, actions, mu, log_var):
        mu = mu.unsqueeze(1)
        log_var = log_var.unsqueeze(1)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + (actions - mu) ** 2 / var + math.log(2 * math.pi))
        return nll.mean()

    def _contrastive(self, x0, actions):
        z = F.normalize(self.align_obs_proj(x0.reshape(x0.shape[0], -1)), dim=-1)
        a = F.normalize(self.align_action_proj(actions.reshape(actions.shape[0], -1)), dim=-1)
        logits = torch.matmul(z, a.transpose(0, 1)) / self.align_temperature
        labels = torch.arange(z.shape[0], device=z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def compute_loss(self, batch):
        obs, instr = self._split_instr(batch['obs'])
        nobs = self.normalizer.normalize(obs)
        nactions = self.normalizer['action'].normalize(batch['action'])
        if nactions.ndim == 4:
            nactions = nactions.reshape(nactions.shape[0], nactions.shape[1], -1)
        B, T, A = nactions.shape

        mu, log_var, z_obs = self._encode_obs(nobs, instr)
        x0 = self.prior.sample(mu, log_var, T)                    # learned source (B, T, A)

        loss_flow, _ = self.flow_matcher.compute_loss(
            self._flow_velocity, target=nactions.reshape(B, T * A),
            start=x0.reshape(B, T * A), global_cond=z_obs)
        if self.prior_type == "none":
            return loss_flow, {"flow_loss": loss_flow.item()}
        loss_nll = self._gaussian_nll(nactions, mu, log_var)
        loss_align = self._contrastive(x0, nactions)

        loss = loss_flow + self.nll_loss_weight * loss_nll + self.align_loss_weight * loss_align
        return loss, {"flow_loss": loss_flow.item(),
                      "nll_loss": loss_nll.item(),
                      "align_loss": loss_align.item()}

    # ---- inference (learned Gaussian source: x0 = mu + sigma * eps) ----
    @torch.no_grad()
    def predict_action(self, obs_dict):
        """Generate actions from a sampled learned prior, including in eval mode.

        Call ``eval()`` before inference to disable dropout. The source remains
        stochastic; seed PyTorch externally for reproducible evaluations.
        Only the first ``n_obs_steps`` frames condition the prediction, even
        when a complete training window is supplied. With ``instr_mode`` set,
        ``obs_dict["instr"]`` is one raw embedding per batch row, (B, instr_dim).
        """
        obs, instr = self._split_instr(obs_dict)
        nobs = self.normalizer.normalize(obs)
        B = next(iter(nobs.values())).shape[0]
        T, A = self.horizon, self.action_dim

        mu, log_var, z_obs = self._encode_obs(nobs, instr)
        mu, log_var = self._apply_prior_intervention(mu, log_var)
        if self.prior_intervention and self.prior_intervention.get("direct_mu"):
            x = mu.unsqueeze(1).expand(B, T, A).reshape(B, T * A)
        else:
            x0 = self.prior.sample(mu, log_var, T)              # same source distribution as training
            x = self.flow_matcher.sample(
                self._flow_velocity, shape=(B, T * A), device=x0.device,
                num_steps=self.num_sampling_steps, start=x0.reshape(B, T * A), global_cond=z_obs)

        naction_pred = x.reshape(B, T, A)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        action = action_pred[:, start:start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}
