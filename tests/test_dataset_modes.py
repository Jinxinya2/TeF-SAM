import json
from pathlib import Path

import numpy as np
from PIL import Image

from tefsam.data.dataset import MedicalSegmentationDataset


def test_segmentation_dataset_returns_image_mask_only(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
        tmp_path / "images" / "case.png"
    )
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(
        tmp_path / "masks" / "case.png"
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["case"]}), encoding="utf-8")

    dataset = MedicalSegmentationDataset(
        annotation_path=str(split_path),
        root_path=str(tmp_path),
        split="train",
        image_size=[8, 8],
    )
    sample = dataset[0]
    assert "text" not in sample
    assert set(sample) == {
        "sample_id",
        "image",
        "label",
        "image_emb",
        "image_feature",
    }
    assert tuple(sample["image"].shape[-2:]) == (8, 8)
