"""Tests for FLUX.3 klein turbo PDD LoRA (4-step generation)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from unittest.mock import MagicMock

from extensions_built_in.diffusion_models.flux3.src.checkpoints import (
    TURBO_FILENAME,
    TURBO_REPO,
)
from extensions_built_in.diffusion_models.flux3.src.turbo_pdd import (
    Flux3TurboPDD,
    LoRAHook,
    PDDHead,
    jump_schedule,
)
from extensions_built_in.diffusion_models.flux3.src.transformer import (
    Flux3Transformer,
    Flux3TransformerParams,
)
from toolkit.models.v2.resolver import resolve_component_file


def test_jump_schedule():
    # 4 steps: 128 // 4 = 32 (within L_min 16 and L_max 32)
    jumps_4 = jump_schedule(4, 128, 16, 32)
    assert jumps_4 == [0, 32, 64, 96, 128]

    # 8 steps: 128 // 8 = 16 (within L_min 16 and L_max 32)
    jumps_8 = jump_schedule(8, 128, 16, 32)
    assert jumps_8 == [0, 16, 32, 48, 64, 80, 96, 112, 128]

    # Invalid step numbers
    for invalid in [5, 50, 0]:
        threw = False
        try:
            jump_schedule(invalid, 128, 16, 32)
        except ValueError:
            threw = True
        assert threw, f"jump_schedule({invalid}) should have raised ValueError"


def test_turbo_pdd_file_and_loading():
    path = resolve_component_file(
        f"{TURBO_REPO}/{TURBO_FILENAME}",
        folder="loras",
        component="FLUX.3 turbo LoRA",
    )
    assert os.path.exists(path), f"Turbo LoRA file does not exist at {path}"

    pdd = Flux3TurboPDD(path, default_steps=4)
    assert pdd.heads_bank is not None
    assert pdd.heads_bank.shape == (128, 128, 3072)
    assert len(pdd.lora_weights) == 202

    # Check 4-step preparation
    assert pdd.fused_heads is not None
    assert pdd.fused_heads.shape == (4, 128, 3072)
    assert pdd.starts is not None
    assert len(pdd.starts) == 4
    assert len(pdd.step_sigmas) == 5
    assert pdd.step_sigmas[0] == 1.0
    assert pdd.step_sigmas[-1] == 0.0

    # Check 8-step preparation
    pdd.prepare_steps(8)
    assert pdd.fused_heads.shape == (8, 128, 3072)
    assert len(pdd.starts) == 8
    assert len(pdd.step_sigmas) == 9
    assert pdd.step_sigmas[0] == 1.0
    assert pdd.step_sigmas[-1] == 0.0


def test_turbo_pdd_activation_and_cleanup():
    path = resolve_component_file(
        f"{TURBO_REPO}/{TURBO_FILENAME}",
        folder="loras",
        component="FLUX.3 turbo LoRA",
    )
    pdd = Flux3TurboPDD(path, default_steps=4)

    # Lightweight transformer for testing activation/cleanup
    params = Flux3TransformerParams(depth=1, depth_single_blocks=2)
    model = Flux3Transformer(params)

    orig_linear = model.final_layer["image"].linear
    assert isinstance(orig_linear, nn.Linear)
    assert not hasattr(model, "set_pdd_step")

    # Count registered forward hooks before
    hooks_before = sum(len(m._forward_hooks) for m in model.modules())

    with pdd.activate(model, steps=4):
        # Inside context
        assert isinstance(model.final_layer["image"].linear, PDDHead)
        assert hasattr(model, "set_pdd_step")
        model.set_pdd_step(0)
        assert model.final_layer["image"].linear.index == 0
        model.set_pdd_step(3)
        assert model.final_layer["image"].linear.index == 3

        # Hooks should be attached to matching modules
        hooks_inside = sum(len(m._forward_hooks) for m in model.modules())
        assert hooks_inside > hooks_before

    # Outside context: full restoration
    assert model.final_layer["image"].linear is orig_linear
    assert not hasattr(model, "set_pdd_step")
    hooks_after = sum(len(m._forward_hooks) for m in model.modules())
    assert hooks_after == hooks_before


def test_turbo_pdd_multiple_sequential_activations():
    path = resolve_component_file(
        f"{TURBO_REPO}/{TURBO_FILENAME}",
        folder="loras",
        component="FLUX.3 turbo LoRA",
    )
    pdd = Flux3TurboPDD(path, default_steps=4)
    params = Flux3TransformerParams(depth=1, depth_single_blocks=1)
    model = Flux3Transformer(params)
    orig_linear = model.final_layer["image"].linear

    for i in range(3):
        with pdd.activate(model, steps=4):
            assert isinstance(model.final_layer["image"].linear, PDDHead)
            model.set_pdd_step(i)
        assert model.final_layer["image"].linear is orig_linear
        assert not hasattr(model, "set_pdd_step")


def test_flux3_pipeline_sigmas_handling():
    from extensions_built_in.diffusion_models.flux3.src.pipeline import Flux3Pipeline

    mock_model = MagicMock()
    mock_model.device_torch = torch.device("cpu")
    mock_model.torch_dtype = torch.bfloat16
    mock_transformer = MagicMock()
    mock_model.transformer = mock_transformer

    pipeline = Flux3Pipeline(mock_model)

    # Test sigmas with terminal 0.0 does not create duplicate 0.0
    sigmas_with_zero = [1.0, 0.95, 0.85, 0.5, 0.0]

    # Pipeline call setup
    cond_embeds = MagicMock()
    cond_embeds.text_embeds = [torch.zeros(16, 20480, dtype=torch.bfloat16)]

    mock_pred = {"x_image": torch.zeros(1, 16, 128, dtype=torch.bfloat16)}
    mock_transformer.return_value = mock_pred
    mock_model.decode_latents.return_value = torch.zeros(1, 3, 64, 64)

    steps_called = []
    def record_step(idx):
        steps_called.append(idx)
    mock_transformer.set_pdd_step = record_step

    images = pipeline(
        conditional_embeds=cond_embeds,
        unconditional_embeds=None,
        height=64,
        width=64,
        num_inference_steps=4,
        sigmas=sigmas_with_zero,
        guidance_scale=1.0,
    )
    assert len(images) == 1
    # Exactly 4 steps called
    assert steps_called == [0, 1, 2, 3]


def test_flux3_model_generate_single_image_turbo():
    from extensions_built_in.diffusion_models.flux3.flux3 import Flux3Model
    from toolkit.config_modules import ModelConfig, GenerateImageConfig

    model_config = ModelConfig(
        name_or_path="black-forest-labs/flux-3-image-klein-base",
        arch="flux3",
    )
    # Instantiate Flux3Model on CPU without heavy weight loading
    flux_model = Flux3Model(
        device="cpu",
        model_config=model_config,
        dtype="bf16",
    )
    # Attach a lightweight transformer
    params = Flux3TransformerParams(depth=1, depth_single_blocks=1)
    flux_model.model = Flux3Transformer(params).to(torch.bfloat16)
    flux_model.vae = MagicMock()
    flux_model.vae.dtype = torch.bfloat16
    flux_model.vae.device = torch.device("cpu")
    flux_model.vae.decode.return_value = torch.zeros(1, 3, 64, 64)

    pipeline = flux_model.get_generation_pipeline()

    gen_config = GenerateImageConfig(
        prompt="A photo of a cat",
        output_path="/tmp/test.png",
        width=64,
        height=64,
        num_inference_steps=50,  # Default 50 from user config
        guidance_scale=3.5,      # Default 3.5 from user config
    )

    cond_embeds = MagicMock()
    cond_embeds.text_embeds = [torch.zeros(16, 20480, dtype=torch.bfloat16)]

    # Mock pipeline to verify kwargs passed by generate_single_image
    captured_kwargs = {}
    def mock_pipe_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        from PIL import Image
        return [Image.new("RGB", (64, 64))]

    mock_pipeline = MagicMock(side_effect=mock_pipe_call)

    img = flux_model.generate_single_image(
        pipeline=mock_pipeline,
        gen_config=gen_config,
        conditional_embeds=cond_embeds,
        unconditional_embeds=None,
        generator=torch.Generator().manual_seed(42),
        extra={},
    )
    assert img is not None
    # Verify turbo 4 steps and guidance_scale=1.0 override were applied
    assert captured_kwargs["num_inference_steps"] == 4
    assert captured_kwargs["guidance_scale"] == 1.0
    assert captured_kwargs["sigmas"] is not None
    assert len(captured_kwargs["sigmas"]) == 5  # 4 steps -> 5 sigmas
    assert captured_kwargs["sigmas"][0] == 1.0
    assert captured_kwargs["sigmas"][-1] == 0.0


if __name__ == "__main__":
    print("Running test_jump_schedule...")
    test_jump_schedule()
    print("Running test_turbo_pdd_file_and_loading...")
    test_turbo_pdd_file_and_loading()
    print("Running test_turbo_pdd_activation_and_cleanup...")
    test_turbo_pdd_activation_and_cleanup()
    print("Running test_turbo_pdd_multiple_sequential_activations...")
    test_turbo_pdd_multiple_sequential_activations()
    print("Running test_flux3_pipeline_sigmas_handling...")
    test_flux3_pipeline_sigmas_handling()
    print("Running test_flux3_model_generate_single_image_turbo...")
    test_flux3_model_generate_single_image_turbo()
    print("ALL TESTS PASSED!")
