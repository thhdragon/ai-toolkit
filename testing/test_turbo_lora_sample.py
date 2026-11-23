"""Tests for turbo LoRA and manual sigma schedule in sample section of training job setup."""

import os
import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import GenerateImageConfig, ModelConfig, SampleConfig
from toolkit.sample_sigmas import normalize_sample_sigmas
from extensions_built_in.diffusion_models.mageflow.mageflow import MageFlowModel
from extensions_built_in.diffusion_models.mageflow.src.transformer import MageFlow, MageFlowParams


def test_normalize_sample_sigmas_string_and_list():
    # List of floats
    res = normalize_sample_sigmas([1.0, 0.75, 0.5, 0.25])
    assert res == [1.0, 0.75, 0.5, 0.25]

    # Comma-separated string
    res_str = normalize_sample_sigmas("1.0, 0.75, 0.5, 0.25")
    assert res_str == [1.0, 0.75, 0.5, 0.25]

    # Timestep scale > 1 (e.g. 1000, 750, 500, 250)
    res_ts = normalize_sample_sigmas([1000, 750, 500, 250])
    assert res_ts == [1.0, 0.75, 0.5, 0.25]


def test_sample_config_turbo_and_manual_sigmas():
    # 1. Config with turbo_lora_path and manual_sigma_schedule as list
    cfg1 = SampleConfig(
        sampler="flowmatch",
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        manual_sigma_schedule=[1.0, 0.75, 0.5, 0.25],
        prompts=["a photo of a cat", "a photo of a dog"],
    )
    assert cfg1.turbo_lora_path == "models/loras/mage-flow-turbo-4-steps.safetensors"
    assert cfg1.sigmas == [1.0, 0.75, 0.5, 0.25]
    assert len(cfg1.samples) == 2
    assert cfg1.samples[0].turbo_lora_path == "models/loras/mage-flow-turbo-4-steps.safetensors"
    assert cfg1.samples[0].sigmas == [1.0, 0.75, 0.5, 0.25]

    # 2. Config with manual_sigma_schedule as string
    cfg2 = SampleConfig(
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        manual_sigma_schedule="1.0, 0.5, 0.25",
        samples=[
            {"prompt": "sample 1"},
            {"prompt": "sample 2", "manual_sigma_schedule": [1.0, 0.8, 0.4]},
        ],
    )
    assert cfg2.sigmas == [1.0, 0.5, 0.25]
    assert cfg2.samples[0].sigmas == [1.0, 0.5, 0.25]
    assert cfg2.samples[1].sigmas == [1.0, 0.8, 0.4]

    # 3. Config with turbo_lora_path without manual_sigma_schedule
    cfg3 = SampleConfig(
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        sample_steps=4,
    )
    assert cfg3.turbo_lora_path == "models/loras/mage-flow-turbo-4-steps.safetensors"
    assert cfg3.sigmas is None
    assert cfg3.sample_steps == 4


def test_generate_image_config_turbo_and_sigmas():
    gen1 = GenerateImageConfig(
        prompt="test",
        output_path="/tmp/test.png",
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        manual_sigma_schedule=[1.0, 0.5],
    )
    assert gen1.turbo_lora_path == "models/loras/mage-flow-turbo-4-steps.safetensors"
    assert gen1.sigmas == [1.0, 0.5]
    assert gen1.num_inference_steps == 2

    gen2 = GenerateImageConfig(
        prompt="test",
        output_path="/tmp/test.png",
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        num_inference_steps=4,
    )
    assert gen2.turbo_lora_path == "models/loras/mage-flow-turbo-4-steps.safetensors"
    assert gen2.sigmas is None
    assert gen2.num_inference_steps == 4


def test_mageflow_model_turbo_lora_generation():
    model_config = ModelConfig(
        name_or_path="microsoft/Mage-Flow-Base",
        arch="mageflow",
    )
    mage_model = MageFlowModel(
        device="cpu",
        model_config=model_config,
        dtype="bf16",
    )

    # Lightweight transformer matching mageflow dimensions
    params = MageFlowParams(
        in_channels=128,
        out_channels=128,
        context_in_dim=2560,
        hidden_size=3072,
        num_heads=24,
        depth=12,
        axes_dim=[32, 48, 48],
        checkpoint=False,
    )
    transformer = MageFlow(params).to(torch.bfloat16)
    mage_model.model = transformer
    mage_model.transformer = transformer

    # 1. Test sampling with turbo LoRA and manual sigma schedule
    gen_config_manual = GenerateImageConfig(
        prompt="A photo of a dog",
        output_path="/tmp/test_manual.png",
        width=64,
        height=64,
        num_inference_steps=25,
        guidance_scale=5.0,
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
        manual_sigma_schedule=[1.0, 0.75, 0.5, 0.25],
    )

    cond_embeds = MagicMock()
    cond_embeds.text_embeds = [torch.zeros(16, 2560, dtype=torch.bfloat16)]

    captured_kwargs = {}
    hooks_count_during_call = []

    def mock_pipe_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        # Check active forward hooks on the transformer during pipeline call
        hook_count = sum(len(m._forward_hooks) for m in transformer.modules())
        hooks_count_during_call.append(hook_count)
        return [Image.new("RGB", (64, 64))]

    mock_pipeline = MagicMock(side_effect=mock_pipe_call)

    # Count hooks before
    hooks_before = sum(len(m._forward_hooks) for m in transformer.modules())
    assert hooks_before == 0

    img = mage_model.generate_single_image(
        pipeline=mock_pipeline,
        gen_config=gen_config_manual,
        conditional_embeds=cond_embeds,
        unconditional_embeds=None,
        generator=torch.Generator().manual_seed(42),
        extra={},
    )
    assert img is not None

    # Hooks were attached during pipeline call and detached afterward
    assert len(hooks_count_during_call) == 1
    assert hooks_count_during_call[0] == 223
    hooks_after = sum(len(m._forward_hooks) for m in transformer.modules())
    assert hooks_after == 0

    # Verify manual sigmas, steps = 4, guidance_scale = 1.0 (CFG override)
    assert captured_kwargs["num_inference_steps"] == 4
    assert captured_kwargs["guidance_scale"] == 1.0
    assert captured_kwargs["sigmas"] == [1.0, 0.75, 0.5, 0.25]

    # 2. Test sampling with turbo LoRA WITHOUT manual sigma schedule (default 4 steps)
    captured_kwargs.clear()
    hooks_count_during_call.clear()

    gen_config_default_schedule = GenerateImageConfig(
        prompt="A photo of a dog",
        output_path="/tmp/test_default.png",
        width=64,
        height=64,
        num_inference_steps=30,  # User default 30 steps
        guidance_scale=5.0,      # User default 5.0 CFG
        turbo_lora_path="models/loras/mage-flow-turbo-4-steps.safetensors",
    )

    img2 = mage_model.generate_single_image(
        pipeline=mock_pipeline,
        gen_config=gen_config_default_schedule,
        conditional_embeds=cond_embeds,
        unconditional_embeds=None,
        generator=torch.Generator().manual_seed(42),
        extra={},
    )
    assert img2 is not None
    assert captured_kwargs["num_inference_steps"] == 4
    assert captured_kwargs["guidance_scale"] == 1.0
    assert captured_kwargs["sigmas"] is None

    hooks_after2 = sum(len(m._forward_hooks) for m in transformer.modules())
    assert hooks_after2 == 0


def test_mageflow_pipeline_sigmas_handling():
    from extensions_built_in.diffusion_models.mageflow.src.pipeline import MageFlowPipeline
    from unittest.mock import patch

    model_mock = MagicMock()
    model_mock.device_torch = torch.device("cpu")
    model_mock.torch_dtype = torch.float32
    model_mock.vae_scale_factor = 16
    model_mock.model_config.model_kwargs = {"static_shift": 6.0}

    params = MageFlowParams(
        in_channels=128,
        out_channels=128,
        context_in_dim=2560,
        hidden_size=3072,
        num_heads=24,
        depth=1,
        axes_dim=[32, 48, 48],
        checkpoint=False,
    )
    transformer = MageFlow(params).to(torch.float32)
    model_mock.transformer = transformer
    model_mock.decode_latents.return_value = torch.zeros(1, 3, 64, 64)

    pipe = MageFlowPipeline(model_mock)
    cond_embeds = MagicMock()
    cond_embeds.text_embeds = [torch.zeros(4, 2560)]

    # Mock predict_velocity so CPU test doesn't require flash_attn cuda kernel
    with patch("extensions_built_in.diffusion_models.mageflow.src.pipeline.predict_velocity") as mock_pv:
        mock_pv.return_value = torch.zeros(1, 128, 4, 4)
        imgs = pipe(
            conditional_embeds=cond_embeds,
            unconditional_embeds=None,
            height=64,
            width=64,
            guidance_scale=1.0,
            sigmas=[1.0, 0.75, 0.5, 0.25],
        )
        assert len(imgs) == 1
        assert isinstance(imgs[0], Image.Image)
        # 4 steps -> predict_velocity called 4 times
        assert mock_pv.call_count == 4


if __name__ == "__main__":
    print("Running test_normalize_sample_sigmas_string_and_list...")
    test_normalize_sample_sigmas_string_and_list()
    print("Running test_sample_config_turbo_and_manual_sigmas...")
    test_sample_config_turbo_and_manual_sigmas()
    print("Running test_generate_image_config_turbo_and_sigmas...")
    test_generate_image_config_turbo_and_sigmas()
    print("Running test_mageflow_model_turbo_lora_generation...")
    test_mageflow_model_turbo_lora_generation()
    print("Running test_mageflow_pipeline_sigmas_handling...")
    test_mageflow_pipeline_sigmas_handling()
    print("ALL TURBO LORA SAMPLE TESTS PASSED!")
