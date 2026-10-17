"""CPU checks for leap.vla.pi05_flow with a fake pi0.5 (no LeRobot needed).

The fake policy's ``denoise_step`` is a linear velocity field v(x, t) = c + k (x - a) + m t with
c = z - a for a fixed (noise z, actions a) pair: with k = m = 0 the flow is exact in both
directions, and the time-dependent variants check the Euler loops step by step against a copy
of LeRobot's ``euler_integrate``.
"""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from leap.vla import pi05_flow as flow
from leap.vla.pi05_flow import (ACTION_KEY, FEAT_DIM, IMAGE2_KEY, IMAGE_KEY, MASK_KEY, STATE_KEY,
                                TASK_KEY, TOKENS_KEY, Observation, Pi05Bundle, PrefixContext)

B, H, D = 2, 50, 32
A = 7


def reference_euler_integrate(denoise_fn, noise, num_steps):
    """Verbatim copy of lerobot.policies.common.flow_matching.euler_integrate (no RTC)."""
    bsize = noise.shape[0]
    device = noise.device
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        v_t = denoise_fn(x_t, time_tensor)
        x_t = x_t + dt * v_t
    return x_t


class FakeFlowModel:
    """pi0.5-like model whose velocity is v(x, t) = c + k (x - a) + m t."""

    def __init__(self, noise, actions, k=0.0, m=0.0):
        self.a = actions
        self.c = noise - actions
        self.k = k
        self.m = m
        self.calls = []
        self.config = SimpleNamespace(num_inference_steps=10)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        assert timestep.dtype == torch.float32 and tuple(timestep.shape) == (x_t.shape[0],)
        assert x_t.dtype == torch.float32
        self.calls.append((x_t.clone(), timestep.clone()))
        return self.c + self.k * (x_t - self.a) + self.m * timestep[:, None, None]

    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None):
        num_steps = num_steps or self.config.num_inference_steps
        return reference_euler_integrate(
            lambda x, t: self.denoise_step(None, None, x, t), noise, num_steps)


class FakePipeline:
    def __init__(self, steps):
        self.steps = steps
        self.inputs = []

    def __call__(self, batch):
        self.inputs.append(batch)
        return dict(batch, processed=True)


def fake_bundle(model, mean=None, std=None):
    mean = np.linspace(-0.5, 0.5, A, dtype=np.float32) if mean is None else mean
    std = np.linspace(0.2, 1.4, A, dtype=np.float32) if std is None else std
    stats_step = SimpleNamespace(stats={ACTION_KEY: {"mean": mean, "std": std}}, eps=1e-8)
    config = SimpleNamespace(chunk_size=H, max_action_dim=D, num_inference_steps=10, n_action_steps=10,
                             output_features={ACTION_KEY: SimpleNamespace(shape=(A,))})
    policy = SimpleNamespace(model=model, config=config)
    return Pi05Bundle(policy=policy, preprocessor=FakePipeline([stats_step]),
                      postprocessor=FakePipeline([]), config=config, device=torch.device("cpu"))


def fake_ctx(batch_size=B):
    return PrefixContext(prefix_pad_masks=torch.ones(batch_size, 5, dtype=torch.bool), past_key_values=None)


class FlowIntegratorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.noise = torch.randn(B, H, D)
        self.actions = torch.randn(B, H, D)
        self.actions[..., A:] = 0.0

    def test_constant_field_is_exact_in_both_directions(self):
        bundle = fake_bundle(FakeFlowModel(self.noise, self.actions))
        ctx = fake_ctx()
        for num_steps in (1, 10):
            forward = flow.forward_from_noise(bundle, ctx, self.noise, num_steps)
            torch.testing.assert_close(forward, self.actions, rtol=0, atol=2e-6)
            reverse = flow.reverse_actions(bundle, ctx, self.actions, num_steps)
            torch.testing.assert_close(reverse, self.noise, rtol=0, atol=2e-6)
        self.assertEqual(tuple(forward.shape), (B, H, D))
        self.assertEqual(forward.dtype, torch.float32)
        self.assertEqual(tuple(reverse.shape), (B, H, D))
        self.assertEqual(reverse.dtype, torch.float32)

    def test_forward_matches_lerobot_euler_loop_bit_for_bit(self):
        model = FakeFlowModel(self.noise, self.actions, k=0.3, m=0.7)
        bundle = fake_bundle(model)
        z = torch.randn(B, H, D)
        ours = flow.forward_from_noise(bundle, fake_ctx(), z, 10)
        model.calls.clear()
        reference = model.sample_actions(None, None, None, None, noise=z, num_steps=10)
        torch.testing.assert_close(ours, reference, rtol=0, atol=0)
        self.assertFalse(torch.allclose(ours, z))

    def test_time_grids_and_step_signs(self):
        model = FakeFlowModel(self.noise, self.actions, k=0.1, m=0.5)
        bundle = fake_bundle(model)
        ctx = fake_ctx()
        flow.forward_from_noise(bundle, ctx, self.noise, 10)
        forward_times = [float(t[0]) for _, t in model.calls]
        self.assertEqual(forward_times, [np.float32(1.0 + s * (-0.1)) for s in range(10)])
        first_x, first_t = model.calls[0]
        torch.testing.assert_close(first_x, self.noise)
        second_x, _ = model.calls[1]
        v0 = model.c + model.k * (self.noise - model.a) + model.m * first_t[:, None, None]
        torch.testing.assert_close(second_x, self.noise + (-0.1) * v0)

        model.calls.clear()
        flow.reverse_actions(bundle, ctx, self.actions, 10)
        reverse_times = [float(t[0]) for _, t in model.calls]
        self.assertEqual(reverse_times, [np.float32(s * 0.1) for s in range(10)])
        first_x, first_t = model.calls[0]
        torch.testing.assert_close(first_x, self.actions)
        second_x, _ = model.calls[1]
        v0 = model.c + model.k * (self.actions - model.a) + model.m * first_t[:, None, None]
        torch.testing.assert_close(second_x, self.actions + 0.1 * v0)

    def test_round_trips_for_a_nearly_constant_linear_field(self):
        bundle = fake_bundle(FakeFlowModel(self.noise, self.actions, k=1e-5))
        ctx = fake_ctx()
        z = torch.randn(B, H, D)
        a = torch.randn(B, H, D)
        torch.testing.assert_close(flow.reverse_actions(bundle, ctx, flow.forward_from_noise(bundle, ctx, z)),
                                   z, rtol=0, atol=1e-5)
        torch.testing.assert_close(flow.forward_from_noise(bundle, ctx, flow.reverse_actions(bundle, ctx, a)),
                                   a, rtol=0, atol=1e-5)

    def test_velocity_accepts_scalar_and_tensor_times(self):
        model = FakeFlowModel(self.noise, self.actions, m=1.0)
        bundle = fake_bundle(model)
        ctx = fake_ctx()
        from_float = flow.velocity(bundle, ctx, self.noise, 0.3)
        from_scalar_tensor = flow.velocity(bundle, ctx, self.noise, torch.tensor(0.3))
        from_batch_tensor = flow.velocity(bundle, ctx, self.noise, torch.full((B,), 0.3))
        torch.testing.assert_close(from_float, from_scalar_tensor, rtol=0, atol=0)
        torch.testing.assert_close(from_float, from_batch_tensor, rtol=0, atol=0)
        for _, t in model.calls:
            self.assertEqual(t.dtype, torch.float32)
            self.assertEqual(tuple(t.shape), (B,))


class NoiseAndNormalisationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.bundle = fake_bundle(FakeFlowModel(torch.zeros(B, H, D), torch.zeros(B, H, D)))

    def test_pad_noise_keeps_the_first_dims_and_fills_the_rest(self):
        noise7 = torch.randn(B, H, A)
        gen = torch.Generator().manual_seed(3)
        padded = flow.pad_noise(noise7, generator=gen)
        self.assertEqual(tuple(padded.shape), (B, H, D))
        self.assertEqual(padded.dtype, torch.float32)
        torch.testing.assert_close(padded[..., :A], noise7, rtol=0, atol=0)
        tail = padded[..., A:]
        self.assertGreater(float(tail.std()), 0.8)
        self.assertLess(abs(float(tail.mean())), 0.2)
        again = flow.pad_noise(noise7.numpy(), generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(again, padded, rtol=0, atol=0)
        self.assertFalse(torch.equal(flow.pad_noise(noise7)[..., A:], tail))
        torch.testing.assert_close(flow.pad_noise(padded), padded, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            flow.pad_noise(torch.zeros(B, H, D + 1))

    def test_normalize_unnormalize_round_trip_uses_the_pipeline_stats(self):
        actions = torch.randn(B, H, A) * 3 + 1
        mean, std, eps = flow.action_stats(self.bundle)
        self.assertEqual(eps, 1e-8)
        normalised = flow.normalize_actions(self.bundle, actions)
        self.assertEqual(tuple(normalised.shape), (B, H, D))
        self.assertEqual(normalised.dtype, torch.float32)
        torch.testing.assert_close(normalised[..., :A], (actions - mean) / (std + eps))
        self.assertTrue(torch.equal(normalised[..., A:], torch.zeros(B, H, D - A)))
        restored = flow.unnormalize_actions(self.bundle, normalised)
        self.assertEqual(tuple(restored.shape), (B, H, A))
        torch.testing.assert_close(restored, actions, rtol=1e-6, atol=1e-6)
        # numpy input and the post-processor fallback for the stats
        torch.testing.assert_close(flow.normalize_actions(self.bundle, actions.numpy()), normalised)
        fallback = Pi05Bundle(self.bundle.policy, FakePipeline([]), self.bundle.preprocessor,
                              self.bundle.config, self.bundle.device)
        torch.testing.assert_close(flow.normalize_actions(fallback, actions), normalised)
        with self.assertRaises(ValueError):
            flow.action_stats(Pi05Bundle(self.bundle.policy, FakePipeline([]), FakePipeline([]),
                                         self.bundle.config, self.bundle.device))
        with self.assertRaises(ValueError):
            flow.normalize_actions(self.bundle, torch.zeros(B, H, A + 1))

    def test_sample_actions_with_noise_returns_env_actions(self):
        noise = torch.randn(B, H, D)
        actions_norm = torch.randn(B, H, D)
        actions_norm[..., A:] = 0.0
        bundle = fake_bundle(FakeFlowModel(noise, actions_norm))
        ctx = fake_ctx()
        out = flow.sample_actions_with_noise(bundle, ctx, noise)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (B, H, A))
        self.assertEqual(out.dtype, np.float32)
        expected = flow.unnormalize_actions(bundle, actions_norm).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
        torch.manual_seed(5)
        gaussian = flow.sample_actions_with_noise(bundle, ctx, None)
        torch.manual_seed(5)
        expected_noise = torch.normal(mean=0.0, std=1.0, size=(B, H, D), dtype=torch.float32)
        np.testing.assert_allclose(gaussian, flow.sample_actions_with_noise(bundle, ctx, expected_noise),
                                   rtol=1e-5, atol=1e-5)


class BatchBuildingTest(unittest.TestCase):
    def setUp(self):
        self.bundle = fake_bundle(FakeFlowModel(torch.zeros(1, H, D), torch.zeros(1, H, D)))
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        self.wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        self.state = rng.standard_normal(8).astype(np.float32)

    def test_image_to_tensor_mirrors_preprocess_observation_and_libero_flip(self):
        tensor = flow.image_to_tensor(self.image, flip=False)
        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor, torch.from_numpy(self.image).permute(2, 0, 1).float() / 255)
        flipped = flow.image_to_tensor(self.image, flip=True)
        torch.testing.assert_close(flipped, torch.flip(tensor, dims=[1, 2]))
        torch.testing.assert_close(flipped[:, 0, 0], tensor[:, -1, -1])
        with self.assertRaises(ValueError):
            flow.image_to_tensor(self.image.astype(np.float32))

    def test_prepare_batch_builds_the_eval_batch_and_runs_the_preprocessor(self):
        batch = flow.prepare_batch(self.bundle, {"image": self.image, "image2": self.wrist},
                                   self.state, "pick up the black bowl and place it on the plate")
        self.assertTrue(batch["processed"])
        raw = self.bundle.preprocessor.inputs[-1]
        self.assertEqual(set(raw), {IMAGE_KEY, IMAGE2_KEY, STATE_KEY, TASK_KEY})
        self.assertEqual(tuple(raw[IMAGE_KEY].shape), (1, 3, 256, 256))
        self.assertEqual(tuple(raw[IMAGE2_KEY].shape), (1, 3, 256, 256))
        self.assertEqual(raw[IMAGE_KEY].dtype, torch.float32)
        torch.testing.assert_close(raw[IMAGE_KEY][0], flow.image_to_tensor(self.image, flip=True))
        torch.testing.assert_close(raw[IMAGE2_KEY][0], flow.image_to_tensor(self.wrist, flip=True))
        self.assertEqual(tuple(raw[STATE_KEY].shape), (1, 8))
        self.assertEqual(raw[STATE_KEY].dtype, torch.float32)
        torch.testing.assert_close(raw[STATE_KEY][0], torch.from_numpy(self.state))
        self.assertEqual(raw[TASK_KEY], ["pick up the black bowl and place it on the plate"])

    def test_prepare_batches_stacks_and_accepts_camera_aliases(self):
        observations = [
            Observation({"agentview_image": self.image, "robot0_eye_in_hand_image": self.wrist}, self.state, "a"),
            Observation({IMAGE_KEY: self.wrist, IMAGE2_KEY: self.image}, self.state * 2, "b"),
        ]
        flow.prepare_batches(self.bundle, observations, flip_images=False)
        raw = self.bundle.preprocessor.inputs[-1]
        self.assertEqual(tuple(raw[IMAGE_KEY].shape), (2, 3, 256, 256))
        torch.testing.assert_close(raw[IMAGE_KEY][1], flow.image_to_tensor(self.wrist, flip=False))
        torch.testing.assert_close(raw[STATE_KEY][1], torch.from_numpy(self.state * 2))
        self.assertEqual(raw[TASK_KEY], ["a", "b"])
        with self.assertRaises(KeyError):
            flow.prepare_batch(self.bundle, {"image": self.image}, self.state, "x")
        with self.assertRaises(ValueError):
            flow.prepare_batch(self.bundle, {"image": self.image, "image2": self.wrist}, self.state[:7], "x")
        with self.assertRaises(ValueError):
            flow.prepare_batches(self.bundle, [])


def fake_vla_utils_modules():
    """Stand-ins for the two lerobot imports prefix_context makes."""
    vla_utils = types.ModuleType("lerobot.policies.common.vla_utils")
    vla_utils.make_att_2d_masks = lambda pad, att: pad[:, None, :] & pad[:, :, None]
    vla_utils.prepare_attention_masks_4d = lambda m: m[:, None]
    modules = {}
    for name in ("lerobot", "lerobot.policies", "lerobot.policies.common"):
        modules[name] = types.ModuleType(name)
    modules["lerobot.policies.common.vla_utils"] = vla_utils
    return modules


class FakeVisionPolicy:
    """Policy whose image tokens are deterministic functions of the image (4 tokens of 2048)."""

    n_tokens = 4

    def __init__(self):
        self.forward_calls = []
        language_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        paligemma = SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        self.model = SimpleNamespace(
            embed_prefix=self.embed_prefix,
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma, forward=self.forward,
                                                  embed_image=self.embed_image),
        )

    def _preprocess_images(self, batch):
        images = [batch[IMAGE_KEY] * 2 - 1, batch[IMAGE2_KEY] * 2 - 1]
        masks = [torch.ones(images[0].shape[0], dtype=torch.bool), torch.ones(images[0].shape[0], dtype=torch.bool)]
        images.append(torch.ones_like(images[0]) * -1)
        masks.append(torch.zeros_like(masks[0]))
        return images, masks

    def embed_image(self, img):
        base = img.mean(dim=(1, 2, 3))[:, None, None]
        scale = torch.arange(1, self.n_tokens + 1, dtype=torch.float32)[None, :, None]
        return base * scale * torch.ones(1, 1, flow.IMAGE_TOKEN_DIM)

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs = [self.embed_image(img) for img in images]
        pad = [m[:, None].expand(m.shape[0], self.n_tokens) for m in img_masks]
        embs.append(torch.zeros(tokens.shape[0], tokens.shape[1], flow.IMAGE_TOKEN_DIM))
        pad.append(masks)
        embs = torch.cat(embs, dim=1)
        pad = torch.cat(pad, dim=1)
        att = torch.zeros_like(pad)
        return embs, pad, att

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return None, "kv-cache"


class PrefixTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.policy = FakeVisionPolicy()
        self.bundle = Pi05Bundle(policy=self.policy, preprocessor=FakePipeline([]), postprocessor=FakePipeline([]),
                                 config=SimpleNamespace(max_action_dim=D), device=torch.device("cpu"))
        self.batch = {
            IMAGE_KEY: torch.rand(B, 3, 8, 8), IMAGE2_KEY: torch.rand(B, 3, 8, 8),
            STATE_KEY: torch.randn(B, 8), TOKENS_KEY: torch.zeros(B, 6, dtype=torch.long),
            MASK_KEY: torch.cat([torch.ones(B, 4, dtype=torch.bool), torch.zeros(B, 2, dtype=torch.bool)], 1),
        }

    def test_prefix_context_runs_paligemma_once_with_cache(self):
        with patch.dict(sys.modules, fake_vla_utils_modules()):
            ctx = flow.prefix_context(self.bundle, self.batch)
        self.assertEqual(ctx.past_key_values, "kv-cache")
        self.assertEqual(ctx.batch_size, B)
        self.assertEqual(tuple(ctx.prefix_pad_masks.shape), (B, 3 * 4 + 6))
        self.assertEqual(ctx.image_token_counts, [4, 4, 4])
        self.assertEqual(len(ctx.images), 3)
        self.assertEqual(len(self.policy.forward_calls), 1)
        call = self.policy.forward_calls[0]
        self.assertTrue(call["use_cache"])
        self.assertIsNone(call["past_key_values"])
        self.assertIsNone(call["inputs_embeds"][1])
        torch.testing.assert_close(call["inputs_embeds"][0], ctx.prefix_embs)
        self.assertEqual(tuple(call["attention_mask"].shape), (B, 1, 18, 18))
        expected_pos = torch.cumsum(ctx.prefix_pad_masks, dim=1) - 1
        torch.testing.assert_close(call["position_ids"], expected_pos)
        self.assertEqual(
            self.policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation,
            "eager")

    def test_prefix_features_pools_real_cameras_and_appends_state(self):
        feat = flow.prefix_features(self.bundle, self.batch)
        self.assertEqual(tuple(feat.shape), (B, FEAT_DIM))
        self.assertEqual(feat.dtype, torch.float32)
        images, _ = self.policy._preprocess_images(self.batch)
        for cam in range(2):
            expected = self.policy.embed_image(images[cam]).mean(dim=1)
            torch.testing.assert_close(feat[:, cam * 2048:(cam + 1) * 2048], expected)
        torch.testing.assert_close(feat[:, 4096:], self.batch[STATE_KEY])
        with patch.dict(sys.modules, fake_vla_utils_modules()):
            ctx = flow.prefix_context(self.bundle, self.batch)
        torch.testing.assert_close(flow.prefix_features(self.bundle, self.batch, ctx), feat, rtol=0, atol=0)


class LoaderTest(unittest.TestCase):
    def test_load_pi05_overrides_the_config_and_builds_processors_like_lerobot_eval(self):
        config = SimpleNamespace(device="cuda", n_action_steps=50, compile_model=True, gradient_checkpointing=True,
                                 dtype="bfloat16", pretrained_path=None)
        policy = MagicMock()
        policy.config = config
        configs_mod = types.ModuleType("lerobot.configs")
        configs_mod.PreTrainedConfig = SimpleNamespace(from_pretrained=MagicMock(return_value=config))
        factory_mod = types.ModuleType("lerobot.policies.factory")
        factory_mod.make_pre_post_processors = MagicMock(return_value=("pre", "post"))
        modeling_mod = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
        modeling_mod.PI05Policy = SimpleNamespace(from_pretrained=MagicMock(return_value=policy))
        modules = {name: types.ModuleType(name) for name in ("lerobot", "lerobot.policies", "lerobot.policies.pi05")}
        modules.update({"lerobot.configs": configs_mod, "lerobot.policies.factory": factory_mod,
                        "lerobot.policies.pi05.modeling_pi05": modeling_mod})
        with patch.dict(sys.modules, modules):
            bundle = flow.load_pi05("some/checkpoint", device="cpu")
        self.assertIs(bundle.policy, policy)
        self.assertEqual((bundle.preprocessor, bundle.postprocessor), ("pre", "post"))
        self.assertEqual(bundle.device, torch.device("cpu"))
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.n_action_steps, 10)
        self.assertEqual(config.pretrained_path, "some/checkpoint")
        self.assertFalse(config.compile_model)
        self.assertEqual(config.dtype, "bfloat16")
        modeling_mod.PI05Policy.from_pretrained.assert_called_once_with("some/checkpoint", config=config)
        policy.to.assert_called_once_with("cpu")
        policy.eval.assert_called_once_with()
        policy.reset.assert_called_once_with()
        factory_mod.make_pre_post_processors.assert_called_once_with(
            policy_cfg=config, pretrained_path="some/checkpoint",
            preprocessor_overrides={"device_processor": {"device": "cpu"},
                                    "rename_observations_processor": {"rename_map": {}}})


if __name__ == "__main__":
    unittest.main()
