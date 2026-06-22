import json
from pathlib import Path

import pytest

from tefsam.config import load_config
from tefsam.engine.runtime import _selection_split


ROOT = Path(__file__).resolve().parents[1]


def test_qata_release_protocol():
    cfg = load_config(ROOT / "configs/qata_cov19_r15.yaml")
    manifest = json.loads(
        (ROOT / "artifact_manifests/qata_cov19_prism_3x2_r15.json").read_text(
            encoding="utf-8"
        )
    )

    assert cfg.training_ratio == pytest.approx(0.15)
    assert cfg.validation_ratio == pytest.approx(0.15)
    assert cfg.batch_size == 32
    assert cfg.ann_path == "./splits/qata_cov19_split.json"
    assert _selection_split(cfg) == "val"
    assert manifest["grid_rows"] == 3
    assert manifest["grid_cols"] == 2
    assert manifest["num_regions"] == 6
    assert manifest["num_prototypes"] == 16
    assert manifest["num_candidates"] == 4


def test_released_qata_split_counts():
    split_path = ROOT / "splits/qata_cov19_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))

    assert {name: len(sample_ids) for name, sample_ids in split.items()} == {
        "train": 5716,
        "val": 1429,
        "test": 2113,
    }


def test_test_split_cannot_select_checkpoints():
    cfg = load_config(ROOT / "configs/qata_cov19_r15.yaml")
    cfg.validation_split = cfg.test_split
    with pytest.raises(ValueError, match="held-out test"):
        _selection_split(cfg)
