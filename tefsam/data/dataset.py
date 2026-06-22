from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandZoomd,
    Resized,
    ToTensord,
)
from torch.utils.data import Dataset, Subset


class MedicalSegmentationDataset(Dataset):
    """Image/mask dataset used by the public segmentation runtime."""

    def __init__(
        self,
        *,
        annotation_path: str,
        root_path: str,
        split: str,
        image_size,
        augment: bool = False,
        image_dir: str = "images",
        mask_dir: str = "masks",
        image_extension: str = ".png",
        mask_extension: str = ".png",
    ) -> None:
        with Path(annotation_path).open("r", encoding="utf-8") as stream:
            split_payload = json.load(stream)
        if split not in split_payload:
            raise KeyError(f"Split '{split}' is absent from {annotation_path}")

        self.sample_ids = list(split_payload[split])
        self.root_path = Path(root_path)
        self.image_size = tuple(int(value) for value in image_size)
        self.augment = augment
        self.image_dir = image_dir.format(split=split)
        self.mask_dir = mask_dir.format(split=split)
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.transform = self._build_transform()

    def __len__(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def _filename(sample_id: str, extension: str) -> str:
        return sample_id if Path(sample_id).suffix else f"{sample_id}{extension}"

    def _build_transform(self):
        transforms = [
            LoadImaged(["image", "mask"], reader="PILReader", image_only=False),
            EnsureChannelFirstd(["image", "mask"]),
        ]
        if self.augment:
            transforms.append(
                RandZoomd(
                    ["image", "mask"],
                    min_zoom=0.95,
                    max_zoom=1.2,
                    mode=["bicubic", "nearest"],
                    prob=0.1,
                )
            )
        transforms.extend(
            [
                Resized(["image"], spatial_size=self.image_size, mode="bicubic"),
                Resized(["mask"], spatial_size=self.image_size, mode="nearest"),
                NormalizeIntensityd(["image"], channel_wise=True),
                ToTensord(["image", "mask"]),
            ]
        )
        return Compose(transforms)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = str(self.sample_ids[index])
        image_path = self.root_path / self.image_dir / self._filename(
            sample_id, self.image_extension
        )
        mask_path = self.root_path / self.mask_dir / self._filename(
            sample_id, self.mask_extension
        )
        transformed = self.transform(
            {"image": str(image_path), "mask": str(mask_path)}
        )
        return {
            "sample_id": sample_id,
            "image": transformed["image"],
            "label": (transformed["mask"] > 0).to(torch.float32),
            "image_emb": None,
            "image_feature": None,
        }


def dataset_from_config(
    cfg,
    *,
    split: str,
    augment: bool,
) -> MedicalSegmentationDataset:
    return MedicalSegmentationDataset(
        annotation_path=cfg.ann_path,
        root_path=cfg.root_path,
        split=split,
        image_size=cfg.image_size,
        augment=augment,
        image_dir=str(getattr(cfg, "image_dir", "images")),
        mask_dir=str(getattr(cfg, "mask_dir", "masks")),
        image_extension=str(getattr(cfg, "image_extension", ".png")),
        mask_extension=str(getattr(cfg, "mask_extension", ".png")),
    )


def deterministic_subset(dataset: Dataset, ratio: float, seed: int) -> Subset:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("subset ratio must be in (0, 1]")
    size = int(ratio * len(dataset))
    if size < 1:
        raise ValueError(
            f"subset ratio {ratio} selects no samples from {len(dataset)} cases"
        )
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:size].tolist()
    return Subset(dataset, indices)
