import types
import unittest
from unittest import mock

import torch

from extensions_built_in.concept_slider.ConceptSliderTrainer import (
    ConceptSliderPromptPair,
    ConceptSliderTrainer,
    ConceptSliderTrainerConfig,
    _build_shuffled_prompts,
    norm_like_tensor,
)
from toolkit.prompt_utils import PromptEmbeds


class _DummyNetwork:
    def __init__(self):
        self.is_active = True
        self.multiplier = 1.0
        self.weight = torch.nn.Parameter(torch.zeros(1, 4, 4, 4))

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier


class _DummySD:
    def __init__(self, embed_map: dict[str, torch.Tensor]):
        self.embed_map = embed_map
        self.unet = torch.nn.Linear(1, 1)
        self.torch_dtype = torch.float32
        self.encode_calls: list[str] = []
        self.predict_calls: list[dict] = []
        self.network: _DummyNetwork | None = None

    def encode_prompt(self, prompts):
        prompt = prompts[0]
        self.encode_calls.append(prompt)
        return PromptEmbeds(self.embed_map[prompt].clone())

    def predict_noise(
        self,
        latents: torch.Tensor,
        conditional_embeddings: PromptEmbeds,
        timestep: torch.Tensor,
        guidance_scale: float = 1.0,
        guidance_embedding_scale: float = 1.0,
        batch=None,
    ) -> torch.Tensor:
        self.predict_calls.append(
            {
                "batch": latents.shape[0],
                "active": self.network.is_active if self.network else None,
                "multiplier": self.network.multiplier if self.network else None,
            }
        )
        # Map each embedding row (1, D) broadcast over spatial dims (C, H, W)
        emb = conditional_embeddings.text_embeds[:, 0, : latents.shape[1]].view(
            latents.shape[0], latents.shape[1], 1, 1
        )
        out = latents + emb
        if self.network is not None and self.network.is_active:
            out = out + self.network.multiplier * self.network.weight
        return out


def _make_trainer(slider_cfg: dict, embed_map: dict[str, torch.Tensor]):
    trainer = object.__new__(ConceptSliderTrainer)
    trainer.slider = ConceptSliderTrainerConfig(**slider_cfg)
    trainer.prompt_pairs = trainer.slider.prompt_pairs
    trainer.prompt_pair_embeds = []
    trainer.positive_prompt = trainer.slider.positive_prompt
    trainer.positive_prompt_embeds = None
    trainer.negative_prompt = trainer.slider.negative_prompt
    trainer.negative_prompt_embeds = None
    trainer.target_class = trainer.slider.target_class
    trainer.target_class_embeds = None
    trainer.anchor_class = trainer.slider.anchor_class
    trainer.anchor_class_embeds = None
    trainer.is_caching_text_embeddings = False
    trainer.device_torch = torch.device("cpu")
    trainer.train_config = types.SimpleNamespace(dtype="fp32")
    trainer.network = _DummyNetwork()
    trainer.sd = _DummySD(embed_map)
    trainer.sd.network = trainer.network
    return trainer


class ConceptSliderConfigTest(unittest.TestCase):
    def test_single_pair_backward_compatibility(self):
        cfg = ConceptSliderTrainerConfig(
            positive_prompt="young person",
            negative_prompt="old person",
            target_class="person",
            anchor_class="",
        )
        self.assertEqual(
            cfg.prompt_pairs,
            [ConceptSliderPromptPair("young person", "old person")],
        )
        self.assertEqual(cfg.positive_prompt, "young person")
        self.assertEqual(cfg.negative_prompt, "old person")

    def test_prompt_pairs_list_and_aliases(self):
        cfg = ConceptSliderTrainerConfig(
            target_class="person",
            prompt_pairs=[
                {"positive_prompt": "young man", "negative_prompt": "old man"},
                {"positive": "young woman", "negative": "old woman"},
            ],
        )
        self.assertEqual(
            cfg.prompt_pairs,
            [
                ConceptSliderPromptPair("young man", "old man"),
                ConceptSliderPromptPair("young woman", "old woman"),
            ],
        )

    def test_list_valued_positive_negative_prompts_and_validation(self):
        cfg = ConceptSliderTrainerConfig(
            positive_prompt=["young man", "young woman"],
            negative_prompt="old person",
        )
        self.assertEqual(
            cfg.prompt_pairs,
            [
                ConceptSliderPromptPair("young man", "old person"),
                ConceptSliderPromptPair("young woman", "old person"),
            ],
        )

        cfg_broadcast = ConceptSliderTrainerConfig(
            positive_prompt=["a", "b"],
            negative_prompt=["c"],
        )
        self.assertEqual(
            cfg_broadcast.prompt_pairs,
            [
                ConceptSliderPromptPair("a", "c"),
                ConceptSliderPromptPair("b", "c"),
            ],
        )
        with self.assertRaises(ValueError):
            ConceptSliderTrainerConfig(
                positive_prompt=["a", "b"],
                negative_prompt=["c", "d", "e"],
            )
        with self.assertRaises(ValueError):
            ConceptSliderTrainerConfig(prompt_pairs=[])


class ConceptSliderTrainerTest(unittest.TestCase):
    def test_hook_caches_unique_prompts_and_averages_concept_direction(self):
        # Channel 0 is neutral/target offset; channel 1 is shared concept; channel 2 is pair-specific bias
        embed_map = {
            "person": torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
            "anchor": torch.tensor([[[0.5, 0.0, 0.0, 0.0]]]),
            "young man": torch.tensor([[[0.0, 1.0, 3.0, 0.0]]]),
            "old man": torch.tensor([[[0.0, -1.0, -3.0, 0.0]]]),
            "young woman": torch.tensor([[[0.0, 1.0, -3.0, 0.0]]]),
            "old woman": torch.tensor([[[0.0, -1.0, 3.0, 0.0]]]),
        }
        trainer = _make_trainer(
            {
                "guidance_strength": 2.0,
                "anchor_strength": 1.0,
                "target_class": "person",
                "anchor_class": "anchor",
                "prompt_pairs": [
                    {"positive_prompt": "young man", "negative_prompt": "old man"},
                    {"positive_prompt": "young woman", "negative_prompt": "old woman"},
                ],
            },
            embed_map,
        )

        with mock.patch(
            "extensions_built_in.sd_trainer.DiffusionTrainer.DiffusionTrainer.hook_before_train_loop",
            return_value=None,
        ):
            trainer.hook_before_train_loop()

        self.assertEqual(len(trainer.prompt_pair_embeds), 2)
        self.assertEqual(len(trainer.sd.encode_calls), 6)

        torch.manual_seed(0)
        noisy_latents = torch.randn(2, 4, 4, 4)
        timesteps = torch.tensor([500, 500])

        loss = trainer.get_guided_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=PromptEmbeds(torch.zeros(2, 1, 4)),
            match_adapter_assist=False,
            network_weight_list=[1.0, 1.0],
            timesteps=timesteps,
            pred_kwargs={},
            batch=types.SimpleNamespace(),
            noise=torch.zeros_like(noisy_latents),
        )

        # 6 prior embed blocks (target, anchor, 2 pairs * 2) chunked into [4, 2] * batch_size(2) -> [8, 4]
        no_grad_batches = [c["batch"] for c in trainer.sd.predict_calls if not c["active"]]
        self.assertEqual(no_grad_batches, [8, 4])

        # Verify expected pure concept direction (bias in channel 2 cancels completely!)
        neutral_pred = noisy_latents
        pure_pos_dir = torch.zeros_like(noisy_latents)
        pure_pos_dir[:, 1, :, :] = 2.0  # (1.0 - (-1.0)) = 2.0
        pure_neg_dir = -pure_pos_dir

        enhance_pos = norm_like_tensor(neutral_pred + 2.0 * pure_pos_dir, neutral_pred)
        erase_neg = norm_like_tensor(neutral_pred - 2.0 * pure_neg_dir, neutral_pred)
        enhance_neg = norm_like_tensor(neutral_pred + 2.0 * pure_neg_dir, neutral_pred)
        erase_pos = norm_like_tensor(neutral_pred - 2.0 * pure_pos_dir, neutral_pred)

        # With initial network.weight == 0, class_pred == neutral_pred and anchor_pred == anchor_target
        expected_pos_loss = (
            torch.nn.functional.mse_loss(neutral_pred, enhance_pos)
            + torch.nn.functional.mse_loss(neutral_pred, erase_neg)
        ) / 3.0
        expected_neg_loss = (
            torch.nn.functional.mse_loss(neutral_pred, enhance_neg)
            + torch.nn.functional.mse_loss(neutral_pred, erase_pos)
        ) / 3.0
        expected_total = (expected_pos_loss + expected_neg_loss) / 2.0

        self.assertAlmostEqual(loss.item(), expected_total.item(), places=5)
        self.assertIsNotNone(trainer.network.weight.grad)
        # Channel 1 gradient should be non-zero along the isolated concept direction
        self.assertGreater(
            trainer.network.weight.grad[:, 1, :, :].abs().mean().item(),
            0.01,
        )
        self.assertEqual(trainer.network.multiplier, 1.0)

        # Compare against a single unbiased pair with the exact isolated direction:
        # loss and gradients must match closely because the opposing biases cancel out.
        pure_trainer = _make_trainer(
            {
                "guidance_strength": 2.0,
                "anchor_strength": 1.0,
                "target_class": "person",
                "anchor_class": "anchor",
                "positive_prompt": "pure_pos",
                "negative_prompt": "pure_neg",
            },
            {
                "person": embed_map["person"],
                "anchor": embed_map["anchor"],
                "pure_pos": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
                "pure_neg": torch.tensor([[[0.0, -1.0, 0.0, 0.0]]]),
            },
        )
        with mock.patch(
            "extensions_built_in.sd_trainer.DiffusionTrainer.DiffusionTrainer.hook_before_train_loop",
            return_value=None,
        ):
            pure_trainer.hook_before_train_loop()

        pure_loss = pure_trainer.get_guided_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=PromptEmbeds(torch.zeros(2, 1, 4)),
            match_adapter_assist=False,
            network_weight_list=[1.0, 1.0],
            timesteps=timesteps,
            pred_kwargs={},
            batch=types.SimpleNamespace(),
            noise=torch.zeros_like(noisy_latents),
        )
        self.assertAlmostEqual(loss.item(), pure_loss.item(), places=5)
        self.assertTrue(
            torch.allclose(
                trainer.network.weight.grad,
                pure_trainer.network.weight.grad,
                atol=1e-6,
            )
        )

    def test_single_pair_without_anchor(self):
        embed_map = {
            "person": torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
            "young person": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
            "old person": torch.tensor([[[0.0, -1.0, 0.0, 0.0]]]),
        }
        trainer = _make_trainer(
            {
                "guidance_strength": 3.0,
                "target_class": "person",
                "positive_prompt": "young person",
                "negative_prompt": "old person",
            },
            embed_map,
        )
        with mock.patch(
            "extensions_built_in.sd_trainer.DiffusionTrainer.DiffusionTrainer.hook_before_train_loop",
            return_value=None,
        ):
            trainer.hook_before_train_loop()

        self.assertIsNone(trainer.anchor_class_embeds)
        loss = trainer.get_guided_loss(
            noisy_latents=torch.randn(1, 4, 4, 4),
            conditional_embeds=PromptEmbeds(torch.zeros(1, 1, 4)),
            match_adapter_assist=False,
            network_weight_list=[1.0],
            timesteps=torch.tensor([500]),
            pred_kwargs={},
            batch=types.SimpleNamespace(),
            noise=torch.zeros(1, 4, 4, 4),
        )
        self.assertEqual([c["batch"] for c in trainer.sd.predict_calls], [3, 1, 1])
        self.assertGreater(loss.item(), 0.0)

    def test_per_pair_shuffle_toggles_tag_shuffling_and_preserves_prose(self):
        prose_pos = "a young person, smiling warmly, sitting near a window"
        prose_neg = "an old person, looking tired, sitting near a window"
        tag_pos = "young, smiling, energetic"
        tag_neg = "old, frowning, tired"

        cfg = ConceptSliderTrainerConfig(
            target_class="person",
            max_permutations=4,
            prompt_pairs=[
                {"positive_prompt": prose_pos, "negative_prompt": prose_neg, "shuffle": False},
                {"positive_prompt": tag_pos, "negative_prompt": tag_neg, "shuffle": True},
            ],
        )
        self.assertFalse(cfg.prompt_pairs[0].shuffle)
        self.assertTrue(cfg.prompt_pairs[1].shuffle)

        # Shuffled variants split on commas and respect max_permutations
        variants = _build_shuffled_prompts("a, b, c, d", max_permutations=4)
        self.assertEqual(len(variants), 4)
        self.assertEqual(len(set(variants)), 4)
        for v in variants:
            self.assertEqual(sorted(t.strip() for t in v.split(",")), ["a", "b", "c", "d"])

        class _RecordingSD(_DummySD):
            def encode_prompt(self, prompts):
                prompt = prompts[0]
                self.encode_calls.append(prompt)
                return PromptEmbeds(torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]))

        trainer = _make_trainer(
            {
                "target_class": "person",
                "max_permutations": 4,
                "prompt_pairs": [
                    {"positive_prompt": prose_pos, "negative_prompt": prose_neg, "shuffle": False},
                    {"positive_prompt": tag_pos, "negative_prompt": tag_neg, "shuffle": True},
                ],
            },
            {},
        )
        trainer.sd = _RecordingSD({})
        trainer.sd.network = trainer.network

        with mock.patch(
            "extensions_built_in.sd_trainer.DiffusionTrainer.DiffusionTrainer.hook_before_train_loop",
            return_value=None,
        ):
            trainer.hook_before_train_loop()

        # Prose pair stayed verbatim (never split on commas)
        self.assertIn(prose_pos, trainer.sd.encode_calls)
        self.assertIn(prose_neg, trainer.sd.encode_calls)
        self.assertEqual(len(trainer.prompt_pair_embeds[0][0]), 1)
        self.assertEqual(len(trainer.prompt_pair_embeds[0][1]), 1)

        # Tag pair generated max_permutations=4 shuffled variants for both positive & negative
        self.assertEqual(len(trainer.prompt_pair_embeds[1][0]), 4)
        self.assertEqual(len(trainer.prompt_pair_embeds[1][1]), 4)
        tag_pos_encoded = [
            p for p in trainer.sd.encode_calls
            if sorted(t.strip() for t in p.split(",")) == ["energetic", "smiling", "young"]
        ]
        self.assertEqual(len(tag_pos_encoded), 4)
        # Verify loss step works when sampling from mixed verbatim + shuffled pairs
        loss = trainer.get_guided_loss(
            noisy_latents=torch.randn(1, 4, 4, 4),
            conditional_embeds=PromptEmbeds(torch.zeros(1, 1, 4)),
            match_adapter_assist=False,
            network_weight_list=[1.0],
            timesteps=torch.tensor([500]),
            pred_kwargs={},
            batch=types.SimpleNamespace(),
            noise=torch.zeros(1, 4, 4, 4),
        )
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
