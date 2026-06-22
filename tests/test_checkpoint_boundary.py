from pathlib import Path

import torch
import torch.nn as nn

from tefsam.engine.checkpoint import (
    CHECKPOINT_FORMAT,
    load_segmentation_checkpoint,
    save_training_checkpoint,
    segmentation_state_dict,
)


class FrozenRepository(nn.Module):
    def __init__(self, artifact_id: str):
        super().__init__()
        self.manifest = {
            "artifact_id": artifact_id,
            "runtime_format": "tefsam-frozen-prism-v1",
            "query_schema": "hierarchical-regional-retrieval-v1",
            "private_path": "/root/private/repository.pt",
        }
        self.register_buffer("matrix", torch.eye(2))


class Core(nn.Module):
    def __init__(self, artifact_id: str = "qata-public-v1"):
        super().__init__()
        self.prototype = FrozenRepository(artifact_id)
        self.head = nn.Linear(2, 1)


class Task(nn.Module):
    def __init__(self, artifact_id: str = "qata-public-v1"):
        super().__init__()
        self.model = Core(artifact_id)


def test_segmentation_checkpoint_excludes_repository():
    state = segmentation_state_dict(Task())
    assert "model.head.weight" in state
    assert not any(key.startswith("model.prototype.") for key in state)


def _save(path: Path, model: nn.Module) -> None:
    save_training_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=4,
        metrics={"dice": 0.8},
        cfg={
            "config_path": "/root/private/config.yaml",
            "repository_config": "/root/private/prism.yaml",
        },
        repository_path="/root/private/repository.pt",
    )


def test_saved_checkpoint_contains_only_public_repository_metadata(tmp_path: Path):
    path = tmp_path / "segmenter.ckpt"
    _save(path, Task())

    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["checkpoint_format"] == CHECKPOINT_FORMAT
    assert "config" not in checkpoint
    assert checkpoint["repository"] == {
        "artifact_id": "qata-public-v1",
        "runtime_format": "tefsam-frozen-prism-v1",
        "query_schema": "hierarchical-regional-retrieval-v1",
    }
    assert "path" not in checkpoint["repository"]
    assert "/root/private" not in repr(checkpoint)


def test_checkpoint_load_does_not_restrict_repository_id(tmp_path: Path):
    path = tmp_path / "segmenter.ckpt"
    source = Task("artifact-a")
    _save(path, source)

    destination = Task("artifact-b")
    load_segmentation_checkpoint(destination, path)

    assert torch.equal(destination.model.head.weight, source.model.head.weight)


def test_legacy_checkpoint_without_artifact_id_remains_loadable(tmp_path: Path):
    path = tmp_path / "legacy.ckpt"
    source = Task("legacy-source")
    torch.save(
        {
            "state_dict": segmentation_state_dict(source),
            "repository": {
                "path": "/root/legacy/repository.pt",
            },
        },
        path,
    )

    destination = Task("current-public-artifact")
    load_segmentation_checkpoint(destination, path)

    assert torch.equal(destination.model.head.weight, source.model.head.weight)
