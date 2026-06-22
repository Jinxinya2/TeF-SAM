from typing import Optional, Protocol

import torch


class SemanticMemory(Protocol):
    """External prototype-memory interface used by TeF-SAM.

    The memory implementation is intentionally not included in the open-source
    model package. Any private or released memory module only needs to implement
    this query method.
    """

    def query(
        self,
        image: torch.Tensor,
        image_emb: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return semantic tokens with shape [B, K, D] or [B, T, D]."""
        ...
