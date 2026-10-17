"""CPU checks for the instruction entry points (instr_mode none/prior/cond/both)."""
from pathlib import Path
import tempfile
import unittest

import torch

from leap import LeaPPolicy
from leap.robotwin.checkpoint import load_policy, save_checkpoint

SHAPE_META = {"obs": {"point_cloud": {"shape": [16, 3]}, "agent_pos": {"shape": [4]}},
              "action": {"shape": [2]}}
SMALL = dict(horizon=8, n_obs_steps=3, n_action_steps=6, encoder_output_dim=16,
             encoder_hidden_dims=[16, 16], flow_net_hidden_dim=32, flow_net_num_layers=1,
             flow_net_time_embed_dim=16)
INSTR = dict(instr_dim=24, instr_adapter_hidden=12, instr_feature_dim=8)


def make_policy(seed=0, **overrides):
    torch.manual_seed(seed)
    return LeaPPolicy(shape_meta=SHAPE_META, **dict(SMALL, **overrides))


class InstrPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.obs = {"point_cloud": torch.randn(2, 3, 16, 3), "agent_pos": torch.randn(2, 3, 4)}
        self.actions = torch.randn(2, 8, 2)
        self.instr = torch.randn(2, 24)

    def fit(self, policy):
        policy.normalizer.fit(dict(self.obs, action=self.actions))
        return policy

    def obs_with_instr(self, instr=None):
        return dict(self.obs, instr=self.instr if instr is None else instr)

    def test_default_mode_is_byte_identical_to_plain_policy(self):
        plain = self.fit(make_policy())
        explicit = self.fit(make_policy(instr_mode="none", **INSTR))
        self.assertEqual(list(plain.state_dict()), list(explicit.state_dict()))
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(explicit.state_dict()[key], value, rtol=0, atol=0)
        self.assertFalse(any("instr" in key for key in plain.state_dict()))
        self.assertIsNone(explicit.instr_adapter)
        self.assertEqual(explicit.flow_net.condition_dim, explicit.obs_feature_dim)
        # No "instr" needed, and a stray "instr" key is ignored rather than normalised.
        explicit.train()
        loss, metrics = explicit.compute_loss({"obs": self.obs, "action": self.actions})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"flow_loss", "nll_loss", "align_loss"})
        explicit.eval()
        torch.manual_seed(3)
        without = explicit.predict_action(self.obs)["action_pred"]
        torch.manual_seed(3)
        with_stray = explicit.predict_action(self.obs_with_instr())["action_pred"]
        torch.testing.assert_close(with_stray, without, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            explicit.set_instr_stats(torch.zeros(24), torch.ones(24))

    def test_each_mode_trains_and_predicts(self):
        for mode in ("prior", "cond", "both"):
            policy = self.fit(make_policy(instr_mode=mode, **INSTR))
            self.assertEqual(policy.instr_mean.shape, (24,))
            self.assertTrue(torch.equal(policy.instr_std, torch.ones(24)))
            policy.train()
            loss, metrics = policy.compute_loss({"obs": self.obs_with_instr(), "action": self.actions})
            self.assertTrue(torch.isfinite(loss), mode)
            self.assertEqual(set(metrics), {"flow_loss", "nll_loss", "align_loss"})
            loss.backward()
            grads = [p.grad for p in policy.instr_adapter.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), mode)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, mode)
            policy.eval()
            result = policy.predict_action(self.obs_with_instr())
            self.assertEqual(result["action"].shape, (2, 6, 2))
            self.assertEqual(result["action_pred"].shape, (2, 8, 2))
            self.assertTrue(torch.isfinite(result["action_pred"]).all())
            with self.assertRaises(ValueError):
                policy.compute_loss({"obs": self.obs, "action": self.actions})
            with self.assertRaises(ValueError):
                policy.predict_action(self.obs)
            with self.assertRaises(ValueError):
                policy.predict_action(self.obs_with_instr(torch.randn(2, 1, 24)))
            with self.assertRaises(ValueError):
                policy.predict_action(self.obs_with_instr(torch.randn(2, 25)))

    def test_prior_mode_widens_only_the_prior_head(self):
        plain = make_policy()
        prior = make_policy(instr_mode="prior", **INSTR)
        self.assertEqual(prior.flow_net.condition_dim, plain.flow_net.condition_dim)
        self.assertEqual(prior.prior.head[1].in_features, plain.prior.head[1].in_features + 8)
        self.assertEqual(prior.prior.head[0].normalized_shape, (plain.prior.head[1].in_features + 8,))
        self.assertEqual(prior.flow_net.cond_embed.in_features, plain.flow_net.cond_embed.in_features)

    def test_cond_mode_widens_only_the_velocity_condition(self):
        plain = make_policy()
        cond = make_policy(instr_mode="cond", **INSTR)
        self.assertEqual(cond.flow_net.condition_dim, plain.flow_net.condition_dim + 8)
        self.assertEqual(cond.flow_net.cond_embed.in_features, plain.flow_net.cond_embed.in_features + 8)
        self.assertEqual(cond.prior.head[1].in_features, plain.prior.head[1].in_features)
        both = make_policy(instr_mode="both", **INSTR)
        self.assertEqual(both.flow_net.condition_dim, plain.flow_net.condition_dim + 8)
        self.assertEqual(both.prior.head[1].in_features, plain.prior.head[1].in_features + 8)

    def test_instruction_reaches_the_configured_path(self):
        other = self.instr + 1.0
        for mode in ("prior", "cond", "both"):
            policy = self.fit(make_policy(instr_mode=mode, **INSTR)).eval()
            with torch.no_grad():
                nobs = policy.normalizer.normalize(self.obs)
                mu_a, _, z_a = policy._encode_obs(nobs, self.instr)
                mu_b, _, z_b = policy._encode_obs(nobs, other)
            if mode == "cond":
                torch.testing.assert_close(mu_a, mu_b, rtol=0, atol=0)   # prior stays state-only
            else:
                self.assertFalse(torch.equal(mu_a, mu_b), mode)
            if mode == "prior":
                torch.testing.assert_close(z_a, z_b, rtol=0, atol=0)     # condition unchanged
            else:
                self.assertFalse(torch.equal(z_a, z_b), mode)
            torch.manual_seed(7)
            first = policy.predict_action(self.obs_with_instr())["action_pred"]
            torch.manual_seed(7)
            second = policy.predict_action(self.obs_with_instr(other))["action_pred"]
            self.assertFalse(torch.equal(first, second), mode)

    def test_stats_standardise_the_embedding(self):
        policy = self.fit(make_policy(instr_mode="cond", **INSTR)).eval()
        mean, std = torch.randn(24), torch.rand(24) + 0.5
        policy.set_instr_stats(mean, std)
        torch.testing.assert_close(policy.instr_mean, mean)
        torch.testing.assert_close(policy.instr_std, std)
        with torch.no_grad():
            z_e = policy._encode_instr(self.instr)
            expected = policy.instr_adapter((self.instr - mean) / std)
        torch.testing.assert_close(z_e, expected)
        self.assertEqual(z_e.shape, (2, 8))
        with self.assertRaises(ValueError):
            policy.set_instr_stats(torch.zeros(23), torch.ones(23))
        with self.assertRaises(ValueError):
            policy.set_instr_stats(torch.zeros(24), torch.zeros(24))

    def test_noprior_composes_with_cond_but_not_prior(self):
        with self.assertRaises(ValueError):
            make_policy(prior_type="none", instr_mode="prior", **INSTR)
        with self.assertRaises(ValueError):
            make_policy(prior_type="none", instr_mode="both", **INSTR)
        with self.assertRaises(ValueError):
            make_policy(instr_mode="late", **INSTR)
        policy = self.fit(make_policy(prior_type="none", instr_mode="cond", **INSTR))
        self.assertIsNone(policy.prior.head)
        self.assertEqual(policy.flow_net.condition_dim, policy.obs_feature_dim + 8)
        policy.train()
        loss, metrics = policy.compute_loss({"obs": self.obs_with_instr(), "action": self.actions})
        self.assertEqual(set(metrics), {"flow_loss"})
        loss.backward()
        policy.eval()
        self.assertEqual(policy.predict_action(self.obs_with_instr())["action"].shape, (2, 6, 2))

    def test_interventions_still_apply_with_instructions(self):
        policy = self.fit(make_policy(instr_mode="both", **INSTR)).eval()
        policy.set_prior_intervention({"mu_override": [0.25, -0.5], "direct_mu": True})
        pred = policy.predict_action(self.obs_with_instr())["action_pred"]
        expected = policy.normalizer["action"].unnormalize(torch.tensor([0.25, -0.5]).expand(2, 8, 2))
        torch.testing.assert_close(pred, expected)
        policy.set_prior_intervention({"sigma_scale": 0})
        torch.manual_seed(1)
        first = policy.predict_action(self.obs_with_instr())["action_pred"]
        torch.manual_seed(2)
        second = policy.predict_action(self.obs_with_instr())["action_pred"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_round_trip_keeps_mode_and_stats(self):
        policy = self.fit(make_policy(instr_mode="both", **INSTR))
        mean, std = torch.randn(24), torch.rand(24) + 0.5
        policy.set_instr_stats(mean, std)
        config = dict(SMALL, shape_meta=SHAPE_META, instr_mode="both", **INSTR)
        optimizer = torch.optim.AdamW(policy.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "both.ckpt"
            save_checkpoint(path, policy, policy, config, optimizer, scheduler, 0, 0)
            loaded = load_policy(path)
        self.assertEqual(loaded.instr_mode, "both")
        torch.testing.assert_close(loaded.instr_mean, mean)
        torch.testing.assert_close(loaded.instr_std, std)
        policy.eval()
        torch.manual_seed(9)
        original = policy.predict_action(self.obs_with_instr())["action_pred"]
        torch.manual_seed(9)
        restored = loaded.predict_action(self.obs_with_instr())["action_pred"]
        torch.testing.assert_close(restored, original, rtol=0, atol=0)

    def test_shape_meta_instr_dim_must_match(self):
        meta = {"obs": dict(SHAPE_META["obs"], instr={"shape": [24]}), "action": SHAPE_META["action"]}
        LeaPPolicy(shape_meta=meta, instr_mode="cond", **dict(SMALL, **INSTR))
        with self.assertRaises(ValueError):
            LeaPPolicy(shape_meta=meta, instr_mode="cond", **dict(SMALL, **dict(INSTR, instr_dim=32)))
        LeaPPolicy(shape_meta=meta, **SMALL)   # instr in shape_meta is irrelevant without instr_mode


if __name__ == "__main__":
    unittest.main()
