from typing import Optional

import torch.nn as nn
from transformers import AutoModel


DEFAULT_CONVNEXT_REVISION = "6166b7613034066690a621d8bf25ffdf181a34f0"


class VisionModel(nn.Module):
    """Return the four hierarchical feature stages of a Hugging Face backbone."""

    def __init__(
        self,
        model_name: str,
        local_files_only: bool = False,
        revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        revision = revision or DEFAULT_CONVNEXT_REVISION
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            output_hidden_states=True,
            local_files_only=local_files_only,
        )
        output_norm = getattr(self.model, "layernorm", None)
        if isinstance(output_norm, nn.Module):
            # The segmenter consumes stage hidden states, not ConvNeXt's pooled head.
            for parameter in output_norm.parameters():
                parameter.requires_grad = False

    def forward(self, image):
        return self.model(image, output_hidden_states=True)["hidden_states"]
