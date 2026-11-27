import itertools
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


@dataclass
class ConceptSliderPromptPair:
    positive_prompt: str = ""
    negative_prompt: str = ""
    shuffle: bool = False


def _build_shuffled_prompts(prompt: str, max_permutations: int = 8) -> list[str]:
    """Split a tag-style prompt on commas and return up to max_permutations unique shuffles."""
    phrases = [phrase.strip() for phrase in prompt.split(",") if phrase.strip()]
    if len(phrases) <= 1:
        return [", ".join(phrases)] if phrases else [prompt]

    limit = max(1, max_permutations)
    if math.factorial(len(phrases)) <= limit:
        variants = list(dict.fromkeys(", ".join(p) for p in itertools.permutations(phrases)))
        random.shuffle(variants)
        return variants[:limit]

    variants: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(variants) < limit and attempts < limit * 4:
        attempts += 1
        shuffled = phrases.copy()
        random.shuffle(shuffled)
        candidate = ", ".join(shuffled)
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    return variants or [", ".join(phrases)]


def _parse_prompt_pair(item, default_shuffle: bool = False) -> ConceptSliderPromptPair:
    if isinstance(item, ConceptSliderPromptPair):
        return item
    if isinstance(item, dict):
        pos = item.get("positive_prompt", item.get("positive", ""))
        neg = item.get("negative_prompt", item.get("negative", ""))
        shuffle = bool(item.get("shuffle", default_shuffle))
        return ConceptSliderPromptPair(
            positive_prompt="" if pos is None else str(pos),
            negative_prompt="" if neg is None else str(neg),
            shuffle=shuffle,
        )
    raise ValueError(
        f"Invalid slider prompt_pairs entry of type {type(item).__name__}; expected dict or ConceptSliderPromptPair."
    )


class ConceptSliderTrainerConfig:
    def __init__(self, **kwargs):
        self.guidance_strength: float = kwargs.get("guidance_strength", 3.0)
        self.anchor_strength: float = kwargs.get("anchor_strength", 1.0)
        self.target_class: str = kwargs.get("target_class", "")
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)
        self.shuffle: bool = bool(kwargs.get("shuffle", False))
        self.max_permutations: int = int(kwargs.get("max_permutations", 8))

        raw_pairs = kwargs.get("prompt_pairs", None)
        raw_positive = kwargs.get("positive_prompt", "")
        raw_negative = kwargs.get("negative_prompt", "")

        if raw_pairs is not None:
            if not isinstance(raw_pairs, (list, tuple)):
                raise ValueError("slider.prompt_pairs must be a list of prompt pair mappings.")
            self.prompt_pairs: list[ConceptSliderPromptPair] = [
                _parse_prompt_pair(item, default_shuffle=self.shuffle) for item in raw_pairs
            ]
            if len(self.prompt_pairs) == 0:
                if raw_positive or raw_negative:
                    self.prompt_pairs = self._pairs_from_pos_neg(
                        raw_positive, raw_negative, shuffle=self.shuffle
                    )
                else:
                    raise ValueError("slider.prompt_pairs must contain at least one prompt pair.")
        else:
            self.prompt_pairs = self._pairs_from_pos_neg(
                raw_positive, raw_negative, shuffle=self.shuffle
            )

        self.positive_prompt: str = self.prompt_pairs[0].positive_prompt
        self.negative_prompt: str = self.prompt_pairs[0].negative_prompt

    @staticmethod
    def _pairs_from_pos_neg(
        raw_positive, raw_negative, shuffle: bool = False
    ) -> list[ConceptSliderPromptPair]:
        if isinstance(raw_positive, (list, tuple)) or isinstance(raw_negative, (list, tuple)):
            pos_list = list(raw_positive) if isinstance(raw_positive, (list, tuple)) else [raw_positive]
            neg_list = list(raw_negative) if isinstance(raw_negative, (list, tuple)) else [raw_negative]
            if len(pos_list) == 1 and len(neg_list) > 1:
                pos_list = pos_list * len(neg_list)
            elif len(neg_list) == 1 and len(pos_list) > 1:
                neg_list = neg_list * len(pos_list)
            elif len(pos_list) != len(neg_list):
                raise ValueError(
                    f"slider.positive_prompt ({len(pos_list)}) and slider.negative_prompt ({len(neg_list)}) must have the same length."
                )
            if len(pos_list) == 0:
                raise ValueError("slider prompt lists must contain at least one prompt pair.")
            return [
                ConceptSliderPromptPair(
                    positive_prompt="" if pos is None else str(pos),
                    negative_prompt="" if neg is None else str(neg),
                    shuffle=shuffle,
                )
                for pos, neg in zip(pos_list, neg_list)
            ]

        return [
            ConceptSliderPromptPair(
                positive_prompt="" if raw_positive is None else str(raw_positive),
                negative_prompt="" if raw_negative is None else str(raw_negative),
                shuffle=shuffle,
            )
        ]


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()
    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean
    return normalized_tensor


class ConceptSliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.do_guided_loss = True

        self.slider: ConceptSliderTrainerConfig = ConceptSliderTrainerConfig(
            **self.config.get("slider", {})
        )

        self.prompt_pairs: list[ConceptSliderPromptPair] = self.slider.prompt_pairs
        self.prompt_pair_embeds: list[tuple[list[PromptEmbeds], list[PromptEmbeds]]] = []
        self.positive_prompt = self.slider.positive_prompt
        self.positive_prompt_embeds: Optional[PromptEmbeds] = None
        self.negative_prompt = self.slider.negative_prompt
        self.negative_prompt_embeds: Optional[PromptEmbeds] = None
        self.target_class = self.slider.target_class
        self.target_class_embeds: Optional[PromptEmbeds] = None
        self.anchor_class = self.slider.anchor_class
        self.anchor_class_embeds: Optional[PromptEmbeds] = None

    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        # cache prompt embeds before super() potentially unloads the text encoder
        with torch.no_grad():
            prompt_cache: dict[str, PromptEmbeds] = {}

            def encode_cached(prompt: str) -> PromptEmbeds:
                if prompt not in prompt_cache:
                    prompt_cache[prompt] = (
                        self.sd.encode_prompt([prompt])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )
                return prompt_cache[prompt]

            self.target_class_embeds = encode_cached(self.target_class)

            if self.anchor_class is not None:
                self.anchor_class_embeds = encode_cached(self.anchor_class)

            self.prompt_pair_embeds = []
            for pair in self.prompt_pairs:
                if pair.shuffle:
                    pos_texts = _build_shuffled_prompts(
                        pair.positive_prompt, self.slider.max_permutations
                    )
                    neg_texts = _build_shuffled_prompts(
                        pair.negative_prompt, self.slider.max_permutations
                    )
                else:
                    pos_texts = [pair.positive_prompt]
                    neg_texts = [pair.negative_prompt]
                self.prompt_pair_embeds.append(
                    (
                        [encode_cached(text) for text in pos_texts],
                        [encode_cached(text) for text in neg_texts],
                    )
                )

            if self.prompt_pair_embeds:
                self.positive_prompt_embeds = self.prompt_pair_embeds[0][0][0]
                self.negative_prompt_embeds = self.prompt_pair_embeds[0][1][0]

        # call parent
        super().hook_before_train_loop()

    def _compute_polarity_loss(
        self,
        multiplier: float,
        noisy_latents: torch.Tensor,
        embeds: PromptEmbeds,
        timesteps: torch.Tensor,
        batch: "DataLoaderBatchDTO",
        enhance_target: torch.Tensor,
        erase_target: torch.Tensor,
        anchor_target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self.network.set_multiplier(multiplier)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if anchor_target is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_target)
        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)

        anchor_loss = anchor_loss * self.slider.anchor_strength

        # send backward now because gradient checkpointing needs network polarity intact
        total_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
        total_loss.backward()
        return total_loss.detach()

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False

        # do our prior preds first
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            self.sd.unet.eval()
            noisy_latents = noisy_latents.to(self.device_torch, dtype=dtype).detach()

            batch_size = noisy_latents.shape[0]

            target_class_embeds = concat_prompt_embeds(
                [self.target_class_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)

            embed_blocks = [target_class_embeds]
            has_anchor = self.anchor_class_embeds is not None
            if has_anchor:
                anchor_embeds = concat_prompt_embeds(
                    [self.anchor_class_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)
                embed_blocks.append(anchor_embeds)

            for pos_variants, neg_variants in self.prompt_pair_embeds:
                pos_embeds = random.choice(pos_variants)
                neg_embeds = random.choice(neg_variants)
                embed_blocks.append(
                    concat_prompt_embeds([pos_embeds] * batch_size).to(
                        self.device_torch, dtype=dtype
                    )
                )
                embed_blocks.append(
                    concat_prompt_embeds([neg_embeds] * batch_size).to(
                        self.device_torch, dtype=dtype
                    )
                )

            # chunk prior predictions to keep peak batch size <= 4 * batch_size
            preds = []
            max_blocks_per_batch = 4
            for start in range(0, len(embed_blocks), max_blocks_per_batch):
                chunk_blocks = embed_blocks[start : start + max_blocks_per_batch]
                num_chunk_blocks = len(chunk_blocks)
                chunk_embeds = concat_prompt_embeds(chunk_blocks)
                chunk_pred = self.sd.predict_noise(
                    latents=torch.cat([noisy_latents] * num_chunk_blocks, dim=0),
                    conditional_embeddings=chunk_embeds,
                    timestep=torch.cat([timesteps] * num_chunk_blocks, dim=0),
                    guidance_scale=1.0,
                    guidance_embedding_scale=1.0,
                    batch=batch,
                )
                preds.extend(chunk_pred.chunk(num_chunk_blocks, dim=0))

            neutral_pred = preds[0]
            if has_anchor:
                anchor_target = preds[1]
                pair_offset = 2
            else:
                anchor_target = None
                pair_offset = 1

            pair_positives = []
            pair_negatives = []
            for pair_idx in range(len(self.prompt_pair_embeds)):
                positive_pred = preds[pair_offset + 2 * pair_idx]
                negative_pred = preds[pair_offset + 2 * pair_idx + 1]
                pair_positives.append(
                    (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
                )
                pair_negatives.append(
                    (negative_pred - neutral_pred) - (positive_pred - neutral_pred)
                )

            # average concept direction across all prompt pairs to isolate the shared concept
            positive = torch.stack(pair_positives, dim=0).mean(dim=0)
            negative = torch.stack(pair_negatives, dim=0).mean(dim=0)

            guidance_scale = self.slider.guidance_strength

            enhance_positive_target = neutral_pred + guidance_scale * positive
            enhance_negative_target = neutral_pred + guidance_scale * negative
            erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive

            # normalize to neutral std/mean
            enhance_positive_target = norm_like_tensor(
                enhance_positive_target, neutral_pred
            )
            enhance_negative_target = norm_like_tensor(
                enhance_negative_target, neutral_pred
            )
            erase_negative_target = norm_like_tensor(
                erase_negative_target, neutral_pred
            )
            erase_positive_target = norm_like_tensor(
                erase_positive_target, neutral_pred
            )

            if was_unet_training:
                self.sd.unet.train()

            # restore network
            if self.network is not None:
                self.network.is_active = was_network_active

            if has_anchor:
                # do a grad inference with our target prompt
                embeds = concat_prompt_embeds([target_class_embeds, anchor_embeds]).to(
                    self.device_torch, dtype=dtype
                )

                noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0).to(
                    self.device_torch, dtype=dtype
                )
                timesteps = torch.cat([timesteps, timesteps], dim=0)
            else:
                embeds = target_class_embeds.to(self.device_torch, dtype=dtype)

        total_pos_loss = self._compute_polarity_loss(
            multiplier=1.0,
            noisy_latents=noisy_latents,
            embeds=embeds,
            timesteps=timesteps,
            batch=batch,
            enhance_target=enhance_positive_target,
            erase_target=erase_negative_target,
            anchor_target=anchor_target,
        )

        total_neg_loss = self._compute_polarity_loss(
            multiplier=-1.0,
            noisy_latents=noisy_latents,
            embeds=embeds,
            timesteps=timesteps,
            batch=batch,
            enhance_target=enhance_negative_target,
            erase_target=erase_positive_target,
            anchor_target=anchor_target,
        )

        self.network.set_multiplier(1.0)

        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # add a grad so backward works right
        total_loss.requires_grad_(True)
        return total_loss
