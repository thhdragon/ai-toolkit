import math
import os
import shutil
import sys
import tempfile
import unittest
from PIL import Image
import torch

# Ensure repo root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.buckets import get_bucket_for_image_size, get_native_resolution_for_image_size
from toolkit.models.base_model import BaseModel
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.dataloader_mixins import BucketsMixin, Bucket
from toolkit.config_modules import DatasetConfig


class DummyMageFlowTransformer(torch.nn.Module):
    """Minimal dummy DiT for testing predict_velocity sequence packing and unpacking."""

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        timesteps: torch.Tensor,
        img_shapes: list,
        img_cu_seqlens: torch.Tensor,
        txt_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # Output shape matches img: [1, total_img_tokens, channels]
        return img


class TestNrMmditBucketing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Create test images with different aspect ratios
        cls.img_landscape = os.path.join(cls.temp_dir, "landscape.png")
        Image.new("RGB", (1200, 800), color="blue").save(cls.img_landscape)

        cls.img_portrait = os.path.join(cls.temp_dir, "portrait.png")
        Image.new("RGB", (600, 900), color="red").save(cls.img_portrait)

        cls.img_square = os.path.join(cls.temp_dir, "square.png")
        Image.new("RGB", (1024, 1024), color="green").save(cls.img_square)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_get_native_resolution_for_image_size(self):
        """Test native resolution calculation preserving aspect ratio without discretization."""
        # 1. Square image within budget
        res = get_native_resolution_for_image_size(512, 512, resolution=512, divisibility=16)
        self.assertEqual(res["width"], 512)
        self.assertEqual(res["height"], 512)

        # 2. Large square image scaled down
        res = get_native_resolution_for_image_size(1024, 1024, resolution=512, divisibility=16)
        self.assertEqual(res["width"], 512)
        self.assertEqual(res["height"], 512)

        # 3. Non-square within budget (does not upscale or crop, just snaps to divisibility)
        res = get_native_resolution_for_image_size(400, 300, resolution=512, divisibility=16)
        self.assertEqual(res["width"] % 16, 0)
        self.assertEqual(res["height"] % 16, 0)
        # 400 is divisible by 16; 300 rounds to 304 (300/16 = 18.75 -> 19 * 16 = 304)
        self.assertEqual(res["width"], 400)
        self.assertEqual(res["height"], 304)

        # 4. Large landscape 16:9 image scaled down preserving native ratio
        res = get_native_resolution_for_image_size(1920, 1080, resolution=512, divisibility=16)
        self.assertEqual(res["width"] % 16, 0)
        self.assertEqual(res["height"] % 16, 0)
        target_area = 512 * 512
        actual_area = res["width"] * res["height"]
        # Area should be close to max budget
        self.assertTrue(abs(actual_area - target_area) / target_area < 0.05)
        # Aspect ratio should closely match 1920 / 1080 (~1.777)
        native_aspect = 1920.0 / 1080.0
        calculated_aspect = res["width"] / res["height"]
        self.assertAlmostEqual(calculated_aspect, native_aspect, delta=0.05)

        # 5. Large portrait 9:16 image scaled down preserving native ratio
        res_portrait = get_native_resolution_for_image_size(1080, 1920, resolution=512, divisibility=16)
        self.assertEqual(res_portrait["width"], res["height"])
        self.assertEqual(res_portrait["height"], res["width"])

        # Contrast with traditional get_bucket_for_image_size:
        bucket = get_bucket_for_image_size(1920, 1080, resolution=512, divisibility=64)
        self.assertEqual(bucket["width"] % 64, 0)
        self.assertEqual(bucket["height"] % 64, 0)

    def test_model_flags(self):
        """Test is_nr_mmdit flag on BaseModel and MageFlow models."""
        self.assertFalse(BaseModel.is_nr_mmdit)

        from extensions_built_in.diffusion_models.mageflow.mageflow import (
            MageFlowModel,
            MageFlowEditModel,
        )
        self.assertTrue(MageFlowModel.is_nr_mmdit)
        self.assertTrue(MageFlowEditModel.is_nr_mmdit)

    def test_file_item_dto_flag(self):
        """Test FileItemDTO preserves is_nr_mmdit attribute."""
        dataset_config = DatasetConfig(dataset_path=self.temp_dir, resolution=512, bucket_tolerance=16)
        dto_default = FileItemDTO(path=self.img_square, dataset_config=dataset_config)
        self.assertFalse(dto_default.is_nr_mmdit)

        dto_nr = FileItemDTO(path=self.img_square, dataset_config=dataset_config, is_nr_mmdit=True)
        self.assertTrue(dto_nr.is_nr_mmdit)

    def test_buckets_mixin_nr_mmdit(self):
        """Test BucketsMixin routes all aspect ratios into a single 'nr' bucket with zero crop."""
        dataset_config = DatasetConfig(
            dataset_path=self.temp_dir,
            resolution=512,
            bucket_tolerance=16,
        )

        class MockDataset(BucketsMixin):
            def __init__(self, items, is_nr_mmdit=False, batch_size=4):
                super().__init__()
                self.dataset_config = dataset_config
                self.dataset_path = "/dummy"
                self.batch_size = batch_size
                self.buckets = {}
                self.batch_indices = []
                self.file_list = items
                self.epoch_num = 0
                self.is_video = False
                self.is_nr_mmdit = is_nr_mmdit

        # Create items for standard diffusion model case (is_nr_mmdit=False):
        std_items = [
            FileItemDTO(path=self.img_landscape, dataset_config=dataset_config, is_nr_mmdit=False),
            FileItemDTO(path=self.img_portrait, dataset_config=dataset_config, is_nr_mmdit=False),
            FileItemDTO(path=self.img_square, dataset_config=dataset_config, is_nr_mmdit=False),
        ]
        std_ds = MockDataset(std_items, is_nr_mmdit=False, batch_size=4)
        std_ds.setup_buckets()
        # Images with different aspect ratios MUST be in different buckets
        self.assertGreater(len(std_ds.buckets), 1)
        for key in std_ds.buckets:
            self.assertNotEqual(key, "nr")

        # Create items for NR-MMDiT model case (is_nr_mmdit=True):
        nr_items = [
            FileItemDTO(path=self.img_landscape, dataset_config=dataset_config, is_nr_mmdit=True),
            FileItemDTO(path=self.img_portrait, dataset_config=dataset_config, is_nr_mmdit=True),
            FileItemDTO(path=self.img_square, dataset_config=dataset_config, is_nr_mmdit=True),
        ]
        nr_ds = MockDataset(nr_items, is_nr_mmdit=True, batch_size=4)
        nr_ds.setup_buckets()

        # All items must be grouped into the single 'nr' bucket
        self.assertEqual(len(nr_ds.buckets), 1)
        self.assertIn("nr", nr_ds.buckets)
        self.assertEqual(len(nr_ds.buckets["nr"].file_list_idx), 3)

        # Verify zero cropping and native resolution scaling
        for item in nr_ds.file_list:
            self.assertEqual(item.crop_x, 0)
            self.assertEqual(item.crop_y, 0)
            self.assertEqual(item.crop_width, item.scale_to_width)
            self.assertEqual(item.crop_height, item.scale_to_height)
            self.assertEqual(item.crop_width % 16, 0)
            self.assertEqual(item.crop_height % 16, 0)

        # Test build_batch_indices:
        # For standard diffusion models, items in a bucket are batched up to batch_size
        std_ds.build_batch_indices()

        # For NR-MMDiT, effective_batch_size is 1 to avoid stacking variable-sized tensors
        nr_ds.build_batch_indices()
        self.assertEqual(len(nr_ds.batch_indices), 3)
        for batch in nr_ds.batch_indices:
            self.assertEqual(len(batch), 1)

    def test_predict_velocity_uniform_tensor(self):
        """Test predict_velocity with a 4D batch tensor."""
        from extensions_built_in.diffusion_models.mageflow.src.pipeline import predict_velocity

        dummy_model = DummyMageFlowTransformer()
        b, c, h, w = 2, 128, 16, 16
        latents = torch.randn(b, c, h, w)
        t = torch.tensor([0.5, 0.5])
        text_embeds = [torch.randn(10, 3072), torch.randn(12, 3072)]

        out = predict_velocity(dummy_model, latents, t, text_embeds)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (b, c, h, w))

    def test_predict_velocity_variable_size_list(self):
        """Test predict_velocity with a list of variable-resolution latents."""
        from extensions_built_in.diffusion_models.mageflow.src.pipeline import predict_velocity

        dummy_model = DummyMageFlowTransformer()
        c = 128
        # Two latents with different native spatial shapes
        latent1 = torch.randn(c, 16, 16)
        latent2 = torch.randn(c, 24, 12)
        t = torch.tensor([0.3, 0.7])
        text_embeds = [torch.randn(8, 3072), torch.randn(14, 3072)]

        out = predict_velocity(dummy_model, [latent1, latent2], t, text_embeds)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (c, 16, 16))
        self.assertEqual(out[1].shape, (c, 24, 12))

    def test_predict_velocity_with_reference_latents(self):
        """Test predict_velocity with reference latents (edit mode)."""
        from extensions_built_in.diffusion_models.mageflow.src.pipeline import predict_velocity

        dummy_model = DummyMageFlowTransformer()
        c = 128
        latent1 = torch.randn(c, 16, 16)
        ref1 = [torch.randn(c, 16, 16)]
        t = torch.tensor([0.5])
        text_embeds = [torch.randn(10, 3072)]

        out = predict_velocity(dummy_model, [latent1], t, text_embeds, ref_latents=[ref1])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (c, 16, 16))


if __name__ == "__main__":
    unittest.main()
