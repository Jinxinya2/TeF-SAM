from unittest.mock import patch

import torch.nn as nn

from tefsam.config import load_config
from tefsam.models.networks.vision import (
    DEFAULT_CONVNEXT_REVISION,
    VisionModel,
)


def test_release_config_pins_convnext_revision():
    cfg = load_config("configs/qata_cov19_r15.yaml")
    assert cfg.vision_revision == DEFAULT_CONVNEXT_REVISION


@patch("tefsam.models.networks.vision.AutoModel.from_pretrained")
def test_vision_model_forwards_revision(from_pretrained):
    backbone = nn.Module()
    backbone.layernorm = nn.LayerNorm(4)
    backbone.stage = nn.Linear(4, 4)
    from_pretrained.return_value = backbone

    model = VisionModel(
        "facebook/convnext-tiny-224",
        revision=DEFAULT_CONVNEXT_REVISION,
        local_files_only=True,
    )

    from_pretrained.assert_called_once_with(
        "facebook/convnext-tiny-224",
        revision=DEFAULT_CONVNEXT_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    assert not any(
        parameter.requires_grad for parameter in model.model.layernorm.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.model.stage.parameters()
    )
