"""CPU checks for the NoPrior baseline and the inference-time prior intervention."""
from pathlib import Path
import tempfile
import unittest

import torch

from leap import LeaPPolicy
from leap.robotwin.checkpoint import load_policy, save_checkpoint


def make_policy(prior_type):
    torch.manual_seed(0)
    return LeaPPolicy(
        shape_meta={"obs": {"point_cloud": {"shape": [16, 3]}, "agent_pos": {"shape": [4]}},
                    "action": {"shape": [2]}},
        horizon=8, n_obs_steps=3, n_action_steps=6,
        encoder_output_dim=16, encoder_hidden_dims=(16, 16),
        flow_net_hidden_dim=32, flow_net_num_layers=1, flow_net_time_embed_dim=16,
        prior_type=prior_type)


class PriorVariantTest(unittest.TestCase):
    def setUp(self):
        self.obs = {"point_cloud": torch.randn(2, 3, 16, 3), "agent_pos": torch.randn(2, 3, 4)}
        self.actions = torch.randn(2, 8, 2)

    def fit(self, policy):
        policy.normalizer.fit(dict(self.obs, action=self.actions))
        return policy

    def test_noprior_has_standard_normal_source_and_flow_loss_only(self):
        policy = self.fit(make_policy("none"))
        self.assertIsNone(policy.prior.head)
        self.assertIsNone(policy.align_obs_proj)
        with torch.no_grad():
            mu, log_var, _ = policy._encode_obs(policy.normalizer.normalize(self.obs))
        self.assertTrue(torch.equal(mu, torch.zeros_like(mu)))
        self.assertTrue(torch.equal(log_var, torch.zeros_like(log_var)))
        torch.manual_seed(1)
        source = policy.prior.sample(mu, log_var, 8)
        torch.manual_seed(1)
        torch.testing.assert_close(source, torch.randn(2, 8, 2))
        policy.train()
        loss, metrics = policy.compute_loss({"obs": self.obs, "action": self.actions})
        self.assertEqual(set(metrics), {"flow_loss"})
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in policy.flow_net.parameters()))
        policy.eval()
        self.assertEqual(policy.predict_action(self.obs)["action"].shape, (2, 6, 2))

    def test_checkpoint_round_trip_keeps_prior_type(self):
        policy = self.fit(make_policy("none"))
        config = dict(shape_meta={"obs": {"point_cloud": {"shape": [16, 3]}, "agent_pos": {"shape": [4]}},
                                  "action": {"shape": [2]}},
                      horizon=8, n_obs_steps=3, n_action_steps=6, encoder_output_dim=16,
                      encoder_hidden_dims=[16, 16], flow_net_hidden_dim=32, flow_net_num_layers=1,
                      flow_net_time_embed_dim=16, prior_type="none")
        optimizer = torch.optim.AdamW(policy.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "none.ckpt"
            save_checkpoint(path, policy, policy, config, optimizer, scheduler, 0, 0)
            loaded = load_policy(path)
            self.assertEqual(loaded.prior_type, "none")
            self.assertIsNone(loaded.prior.head)
            # A checkpoint written before prior_type existed still loads as LeaP.
            del config["prior_type"]
            legacy = self.fit(make_policy("leap"))
            save_checkpoint(Path(tmp) / "legacy.ckpt", legacy, legacy, config, optimizer, scheduler, 0, 0)
            loaded = load_policy(Path(tmp) / "legacy.ckpt")
            self.assertEqual(loaded.prior_type, "leap")
            self.assertIsNotNone(loaded.prior.head)

    def test_unknown_prior_type_rejected(self):
        with self.assertRaises(ValueError):
            make_policy("gmm")

    def test_intervention_none_is_a_noop(self):
        policy = self.fit(make_policy("leap")).eval()
        torch.manual_seed(5)
        base = policy.predict_action(self.obs)["action_pred"]
        policy.set_prior_intervention({"sigma_scale": 1.0, "mu_shift": [0.0, 0.0]})
        torch.manual_seed(5)
        same = policy.predict_action(self.obs)["action_pred"]
        policy.set_prior_intervention(None)
        torch.manual_seed(5)
        restored = policy.predict_action(self.obs)["action_pred"]
        torch.testing.assert_close(same, base, rtol=0, atol=0)
        torch.testing.assert_close(restored, base, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            policy.set_prior_intervention({"temperature": 2})
        with self.assertRaises(ValueError):
            policy.set_prior_intervention({"sigma_scale": -1})
        with self.assertRaises(TypeError):
            policy.set_prior_intervention([1])

    def test_sigma_scale_zero_is_deterministic(self):
        policy = self.fit(make_policy("leap")).eval()
        policy.set_prior_intervention({"sigma_scale": 0})
        torch.manual_seed(1)
        first = policy.predict_action(self.obs)["action_pred"]
        torch.manual_seed(2)
        second = policy.predict_action(self.obs)["action_pred"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())

    def test_direct_mu_and_overrides(self):
        policy = self.fit(make_policy("leap")).eval()
        policy.set_prior_intervention({"mu_override": [0.25, -0.5], "direct_mu": True})
        pred = policy.predict_action(self.obs)["action_pred"]
        expected = policy.normalizer["action"].unnormalize(
            torch.tensor([0.25, -0.5]).expand(2, 8, 2))
        torch.testing.assert_close(pred, expected)
        policy.set_prior_intervention({"logvar_override": [0.0, 0.0], "mu_shift": [1.0, 1.0]})
        with torch.no_grad():
            mu, log_var, _ = policy._encode_obs(policy.normalizer.normalize(self.obs))
            mu2, log_var2 = policy._apply_prior_intervention(mu, log_var)
        torch.testing.assert_close(mu2, mu + 1.0)
        self.assertTrue(torch.equal(log_var2, torch.zeros_like(log_var2)))


if __name__ == "__main__":
    unittest.main()
