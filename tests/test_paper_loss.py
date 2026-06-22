from types import SimpleNamespace

import torch
import torch.nn as nn
from monai.losses import DiceLoss

import tefsam.models.task as task_module
from tefsam.models.task import TeFSAM


class FixedSegmenter(nn.Module):
    def __init__(self, cfg, prototype):
        super().__init__()
        del cfg, prototype
        self.final_logits = nn.Parameter(
            torch.tensor([[[[-2.0, 1.0], [0.5, 3.0]]]])
        )
        self.coarse_logits = nn.Parameter(
            torch.tensor([[[[-1.0, 2.0], [1.5, -0.5]]]])
        )

    def forward(self, image, **kwargs):
        del image, kwargs
        return {
            "logits": self.final_logits,
            "aux_logits": self.coarse_logits,
        }


def test_equation_21_uses_main_dice_ce_and_coarse_ce(monkeypatch):
    monkeypatch.setattr(task_module, "TeFSAMSegmenter", FixedSegmenter)
    model = TeFSAM(SimpleNamespace(mask_prompt_loss_weight=0.2), nn.Identity())
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    output = model({"image": torch.zeros(1, 1, 2, 2), "label": target})

    expected_dice = DiceLoss(sigmoid=True)(model.model.final_logits, target)
    expected_ce = nn.BCEWithLogitsLoss()(model.model.final_logits, target)
    expected_main = expected_dice + expected_ce
    expected_coarse = nn.BCEWithLogitsLoss()(model.model.coarse_logits, target)
    expected = expected_main + 0.2 * expected_coarse

    assert torch.allclose(output["loss"], expected)
    assert torch.allclose(output["loss_dice"], expected_dice)
    assert torch.allclose(output["loss_ce"], expected_ce)
    assert expected_ce.item() > 0.0
    assert torch.equal(output["logits"], model.model.final_logits)

    output["loss"].backward()
    assert model.model.final_logits.grad is not None
    assert model.model.final_logits.grad.abs().sum().item() > 0.0
    assert model.model.coarse_logits.grad is not None
    assert model.model.coarse_logits.grad.abs().sum().item() > 0.0
