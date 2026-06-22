from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from transformers import AutoModel, AutoTokenizer


class VisionPyramidEncoder(nn.Module):
    """Returns four visual feature maps ordered from high to low resolution."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        states = self.model(image, output_hidden_states=True)["hidden_states"]
        maps = [x for x in states if x.dim() == 4]
        if len(maps) < 4:
            raise RuntimeError("Vision encoder must expose at least four 2D hidden-state maps.")
        return maps[-4:]


class BiomedCLIPEncoder(nn.Module):
    """Image/text encoder used to build and query the prototype memory."""

    def __init__(self, model_path: str, tokenizer_name: str, device: str) -> None:
        super().__init__()
        self.device_name = device
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def _ensure_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 1:
            image = repeat(image, "b 1 h w -> b 3 h w")
        return image

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self._ensure_rgb(image).to(next(self.model.parameters()).device)
        output = self.model.vision_model(image)
        emb = output[1] if isinstance(output, (tuple, list)) else output.pooler_output
        return F.normalize(emb, dim=-1)

    def encode_text(self, text: List[str], max_length: int = 24) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(next(self.model.parameters()).device) for k, v in tokens.items()}
        output = self.model.text_model(tokens["input_ids"], tokens["attention_mask"])
        emb = output[1] if isinstance(output, (tuple, list)) else output.pooler_output
        return F.normalize(emb, dim=-1)

    def encode_text_feature(self, text: List[str], max_length: int = 256) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(next(self.model.parameters()).device) for k, v in tokens.items()}
        output = self.model.text_model(tokens["input_ids"], tokens["attention_mask"])
        return output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
