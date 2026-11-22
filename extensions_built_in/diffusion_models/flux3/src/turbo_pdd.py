"""FLUX.3 klein turbo PDD (Parallel Decoding Distillation) student support.

Accelerates sample generation for FLUX.3 klein base from 50 steps down to
4 steps (or 8 steps) using NoiseToNova/Flux-3-Klein-turbo-pdd-4_8-steps.

The turbo LoRA combines:
  - Multi-step output heads (trajectory_steps=128 on a shifted training grid):
    for a jump schedule (e.g. 4 steps = jumps of 32 steps on the 128-step grid),
    the multi-step heads are fused via a dt-weighted average, replacing
    final_layer.image.linear.
  - Rank-128 LoRA weights across content_mode_blocks, single_blocks,
    modulations, and embeddings (202 target linear projections).
  - Distilled classifier-free guidance (teacher CFG 4 distilled into student),
    enabling sampling with guidance_scale=1.0 (a single forward pass per step).
"""

from contextlib import contextmanager
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .checkpoints import TURBO_FILENAME, TURBO_REPO

DEFAULT_PDD_CONFIG = {
    "trajectory_steps": 128,
    "L_min": 16,
    "L_max": 32,
    "shift": 6.93,
    "guidance_scale": 4,
    "lora_rank": 128,
    "lora_alpha": 128,
}


def jump_schedule(
    steps: int,
    trajectory_steps: int = 128,
    L_min: int = 16,
    L_max: int = 32,
) -> List[int]:
    """Grid indices of `steps` jumps across `trajectory_steps` steps.

    Each jump block size must be a multiple of L_min and at most L_max.
    For 4 steps on 128 grid: block_size = 32, boundaries = [0, 32, 64, 96, 128].
    For 8 steps on 128 grid: block_size = 16, boundaries = [0, 16, 32, ..., 128].
    """
    block = trajectory_steps // steps if steps > 0 else 0
    if (
        steps < 1
        or trajectory_steps % steps != 0
        or block % L_min != 0
        or not (L_min <= block <= L_max)
    ):
        raise ValueError(
            f"{steps} steps do not fit {trajectory_steps} grid steps with jumps of {L_min}..{L_max} "
            f"(multiples of {L_min})"
        )
    return list(range(0, trajectory_steps + 1, block))


class PDDHead(nn.Module):
    """Multi-step fused head replacing final_layer.image.linear during turbo sampling."""

    def __init__(
        self,
        heads: torch.Tensor,
        orig_linear: nn.Linear,
        starts: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.heads = heads  # (jumps, out_channels, in_channels) fp32 on CPU
        self.orig_linear = orig_linear
        self.starts = starts  # (jumps,) starting sigma values
        self.index = 0
        self._cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def set_step(self, step_idx: int) -> None:
        self.index = min(max(0, step_idx), len(self.heads) - 1)

    def set_step_by_sigma(self, sigma: float) -> None:
        if self.starts is not None:
            self.index = int((self.starts - sigma).abs().argmin())

    def clear_cache(self) -> None:
        self._cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = self.heads.to(device=x.device, dtype=x.dtype)
        idx = min(max(0, self.index), len(self.heads) - 1)
        return F.linear(x, self._cache[key][idx])


class LoRAHook:
    """Forward hook adding the LoRA delta to a linear layer's output."""

    def __init__(self, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float = 1.0):
        self.lora_a = lora_a  # (rank, in_dim)
        self.lora_b = lora_b  # (out_dim, rank)
        self.scale = scale
        self._cache_a: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._cache_b: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._cache_a.clear()
        self._cache_b.clear()

    def __call__(self, module: nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        x = args[0]
        key = (x.device, x.dtype)
        if key not in self._cache_a:
            self._cache_a[key] = self.lora_a.to(device=x.device, dtype=x.dtype)
            self._cache_b[key] = self.lora_b.to(device=x.device, dtype=x.dtype)
        delta = F.linear(F.linear(x, self._cache_a[key]), self._cache_b[key]) * self.scale
        return output + delta.to(output.dtype)


class Flux3TurboPDD:
    """Manager for FLUX.3 turbo PDD LoRA weights and sampling schedules."""

    def __init__(self, weight_path: str, default_steps: int = 4):
        self.weight_path = weight_path
        self.default_steps = default_steps
        self.config: dict = {}
        self.heads_bank: Optional[torch.Tensor] = None
        self.lora_weights: Dict[str, dict] = {}
        self.fused_heads: Optional[torch.Tensor] = None
        self.starts: Optional[torch.Tensor] = None
        self.step_sigmas: Optional[List[float]] = None
        self.current_steps: Optional[int] = None
        self._load()

    def _load(self) -> None:
        with safe_open(self.weight_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if "pdd_config" in meta:
                self.config = json.loads(meta["pdd_config"])
            else:
                self.config = dict(DEFAULT_PDD_CONFIG)

            if "pdd.heads" in f.keys():
                self.heads_bank = f.get_tensor("pdd.heads").float()
            elif "proj_out.weight" in f.keys():
                self.heads_bank = f.get_tensor("proj_out.weight").float()
            else:
                raise KeyError(f"No pdd.heads or proj_out.weight found in {self.weight_path}")

            for k in f.keys():
                if k in ("pdd.heads", "proj_out.weight"):
                    continue
                clean_k = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
                clean_k = clean_k[len("transformer."):] if clean_k.startswith("transformer.") else clean_k

                if clean_k.endswith(".lora_A.weight"):
                    mname = clean_k[:-len(".lora_A.weight")]
                    self.lora_weights.setdefault(mname, {})["lora_A"] = f.get_tensor(k)
                elif clean_k.endswith(".lora_B.weight"):
                    mname = clean_k[:-len(".lora_B.weight")]
                    self.lora_weights.setdefault(mname, {})["lora_B"] = f.get_tensor(k)
                elif clean_k.endswith(".alpha"):
                    mname = clean_k[:-len(".alpha")]
                    self.lora_weights.setdefault(mname, {})["alpha"] = f.get_tensor(k)
                # Diffusers PEFT key support: .lora_A.default.weight / .lora_B.default.weight
                elif ".lora_A.default.weight" in clean_k:
                    mname = clean_k.replace(".lora_A.default.weight", "")
                    self.lora_weights.setdefault(mname, {})["lora_A"] = f.get_tensor(k)
                elif ".lora_B.default.weight" in clean_k:
                    mname = clean_k.replace(".lora_B.default.weight", "")
                    self.lora_weights.setdefault(mname, {})["lora_B"] = f.get_tensor(k)

        for mname, w in self.lora_weights.items():
            if "lora_A" in w:
                rank = w["lora_A"].shape[0]
                alpha = float(w.get("alpha", rank))
                w["scale"] = alpha / rank

        self.prepare_steps(self.default_steps)

    def prepare_steps(self, steps: int = 4) -> None:
        if self.current_steps == steps and self.fused_heads is not None:
            return

        trajectory_steps = int(self.config.get("trajectory_steps", 128))
        L_min = int(self.config.get("L_min", 16))
        L_max = int(self.config.get("L_max", 32))
        boundaries = jump_schedule(steps, trajectory_steps, L_min, L_max)

        grid = torch.tensor(self.config["sigmas"], dtype=torch.float64)
        dt = grid.diff()
        fused = []
        for n, m in zip(boundaries[:-1], boundaries[1:]):
            mix = (dt[n:m] / dt[n:m].sum()).float()
            fused.append(torch.einsum("t,toi->oi", mix, self.heads_bank[n:m]))

        self.fused_heads = torch.stack(fused)  # (steps, out_channels, in_channels)
        self.starts = grid[boundaries[:-1]].float()
        self.step_sigmas = grid[boundaries].float().tolist()
        self.current_steps = steps

    @contextmanager
    def activate(self, transformer: nn.Module, steps: int = 4):
        """Temporarily activate turbo LoRA forward hooks and multi-step head."""
        self.prepare_steps(steps)
        modules = dict(transformer.named_modules())
        hook_handles = []
        active_hooks: List[LoRAHook] = []

        # 1. Attach LoRA forward hooks
        for mname, weights in self.lora_weights.items():
            if "lora_A" not in weights or "lora_B" not in weights:
                continue
            if mname in modules:
                hook = LoRAHook(
                    weights["lora_A"],
                    weights["lora_B"],
                    scale=weights.get("scale", 1.0),
                )
                active_hooks.append(hook)
                hook_handles.append(modules[mname].register_forward_hook(hook))

        # 2. Swap final_layer.image.linear with PDDHead
        has_final_layer = (
            hasattr(transformer, "final_layer")
            and "image" in transformer.final_layer
            and hasattr(transformer.final_layer["image"], "linear")
        )
        orig_linear = None
        pdd_head = None
        if has_final_layer:
            orig_linear = transformer.final_layer["image"].linear
            pdd_head = PDDHead(self.fused_heads, orig_linear, starts=self.starts)
            transformer.final_layer["image"].linear = pdd_head
            transformer.set_pdd_step = pdd_head.set_step

        try:
            yield self
        finally:
            for h in hook_handles:
                h.remove()
            for hook in active_hooks:
                hook.clear_cache()
            if has_final_layer and orig_linear is not None:
                transformer.final_layer["image"].linear = orig_linear
                if hasattr(transformer, "set_pdd_step"):
                    delattr(transformer, "set_pdd_step")
            if pdd_head is not None:
                pdd_head.clear_cache()

