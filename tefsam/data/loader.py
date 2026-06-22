from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def collate_optional(batch):
    output = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if all(value is None for value in values):
            output[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"Batch field '{key}' mixes missing and present values")
        else:
            output[key] = torch.utils.data.default_collate(values)
    return output


class MedicalDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, collate_fn=collate_optional, **kwargs)
