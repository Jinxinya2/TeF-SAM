from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import tefsam.models.segmenter as segmenter_module
from tefsam.models.task import TeFSAM


class FakeRepository(nn.Module):
    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.register_buffer("repository_ready", torch.tensor(True))

    def query(self, images, image_emb=None, image_feature=None):
        return images.new_zeros((images.size(0), 2, self.feature_dim))


class FakeVision(nn.Module):
    def __init__(self, _name, **kwargs):
        super().__init__()

    def forward(self, image):
        batch = image.size(0)
        return (
            image.new_zeros((batch, 2, 8, 8)),
            image.new_zeros((batch, 2, 8, 8)),
            image.new_zeros((batch, 4, 4, 4)),
            image.new_zeros((batch, 6, 2, 2)),
            image.new_zeros((batch, 8, 1, 1)),
        )


class FakeFusion(nn.Module):
    def __init__(self, in_channels, prompt_dim, token_grid):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.prompt_dim = prompt_dim

    def forward(self, feature_maps):
        batch = feature_maps[0].size(0)
        tokens = feature_maps[0].new_ones((batch, 4, self.prompt_dim)) * self.scale
        return feature_maps[0], tokens


class FakeSAMAdapter(nn.Module):
    def __init__(self, image_dim, text_dim, prompt_dim, **kwargs):
        super().__init__()
        self.head = nn.Linear(prompt_dim, 1)

    def forward(self, image_tokens, text_tokens, target_size, target=None):
        del text_tokens, target
        batch = image_tokens.size(0)
        logits = self.head(image_tokens).transpose(1, 2).reshape(batch, 1, 2, 2)
        logits = F.interpolate(logits, size=target_size, mode="bilinear")
        return logits, logits


def config():
    return SimpleNamespace(
        agg="memory",
        use_approx1=False,
        vision_type="fake",
        vision_feature_dims=[8, 6, 4, 2],
        vision_spatial_dims=[1, 2, 4, 8],
        text_dim=8,
        num_candidate=4,
        sam_prompt_dim=4,
        sam_image_embedding=2,
        fusion_token_grid=2,
        image_size=[8, 8],
        point_topk=2,
        point_threshold=0.5,
        sam_checkpoint=None,
        sam_model_type="vit_b",
        sam_lora_enabled=False,
        mask_prompt_loss_weight=0.2,
    )


def test_image_and_mask_only_batch_backpropagates(monkeypatch):
    monkeypatch.setattr(segmenter_module, "VisionModel", FakeVision)
    monkeypatch.setattr(segmenter_module, "MultiScaleFeatureFusion", FakeFusion)
    monkeypatch.setattr(segmenter_module, "MedicalSAMAdapter", FakeSAMAdapter)

    model = TeFSAM(config(), FakeRepository())
    batch = {
        "image": torch.randn(2, 1, 8, 8),
        "label": (torch.rand(2, 1, 8, 8) > 0.5).float(),
        "image_emb": None,
        "image_feature": None,
    }
    output = model(batch)
    output["loss"].backward()

    assert output["logits"].shape == batch["label"].shape
    assert model.model.fusion.scale.grad is not None
    assert model.model.sam_adapter.head.weight.grad is not None
