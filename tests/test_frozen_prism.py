import json

import torch
import torch.nn as nn
from safetensors.torch import save_file

import tefsam.models.frozen_prism as frozen_module
from tefsam.models.frozen_prism import FrozenPRISMRuntime


class FakeVisionEncoder(nn.Module):
    def __init__(self, checkpoint, *, revision=None, local_files_only=False):
        super().__init__()
        del checkpoint, revision, local_files_only
        self.projection = nn.Linear(2, 2, bias=False)

    def encode_image_with_tokens(self, image):
        batch = image.size(0)
        pooled = image.new_tensor([[1.0, 0.0]]).expand(batch, -1)
        patches = image.new_tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
        ).expand(batch, -1, -1)
        return self.projection(pooled), self.projection(patches)


def write_runtime_artifact(tmp_path):
    artifact = tmp_path / "runtime.safetensors"
    tensors = {
        "prototype_keys": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
        ),
        "memory_text": torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3),
        "prototype_support": torch.ones(2, 2, dtype=torch.bool),
        "global_image_center": torch.zeros(2),
        "regional_query_center": torch.zeros(2),
        "memory_anchor": torch.zeros(2, 3),
        "memory_residual_scale": torch.ones(()),
        "image_projector.weight": torch.eye(2),
        "region_head.weight": torch.zeros(2, 2),
        "region_head.bias": torch.zeros(2),
        "vision_delta.projection.weight": torch.eye(2),
    }
    save_file(tensors, str(artifact))
    manifest = {
        "manifest_version": 1,
        "runtime_format": "tefsam-frozen-prism-v1",
        "query_schema": "hierarchical-regional-retrieval-v1",
        "artifact_id": "test-prism",
        "artifact_filename": artifact.name,
        "num_regions": 2,
        "num_prototypes": 2,
        "num_candidates": 1,
        "embedding_dim": 2,
        "retrieval_dim": 2,
        "feature_dim": 3,
        "memory_tokens": 2,
        "grid_rows": 1,
        "grid_cols": 2,
        "retrieval_temperature": 0.1,
        "response_temperature": 1.0,
        "patch_mil_temperature": 0.07,
        "region_gate_weight": 1.0,
        "region_position_scale": 0.1,
        "image_encoder": {"checkpoint": "fake", "revision": None},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, manifest_path


def test_load_query_and_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(
        frozen_module, "_BiomedCLIPVisionEncoder", FakeVisionEncoder
    )
    artifact, manifest = write_runtime_artifact(tmp_path)
    runtime = FrozenPRISMRuntime.load(
        artifact, manifest_path=manifest, device="cpu"
    )

    response = runtime.query(torch.zeros(2, 3, 2, 2))

    assert response.shape == (2, 2, 3)
    assert runtime.artifact_id == "test-prism"
    assert bool(runtime.repository_ready.item())
    assert not any(parameter.requires_grad for parameter in runtime.parameters())
    runtime.train()
    assert not runtime.training
    assert not runtime.image_encoder.training


def test_artifact_filename_and_id_are_not_restricted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        frozen_module, "_BiomedCLIPVisionEncoder", FakeVisionEncoder
    )
    artifact, manifest = write_runtime_artifact(tmp_path)
    renamed_artifact = tmp_path / "renamed-memory.safetensors"
    artifact.rename(renamed_artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("artifact_id")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runtime = FrozenPRISMRuntime.load(renamed_artifact, manifest_path=manifest)

    assert runtime.artifact_id is None


def test_local_base_checkpoint_directory_is_not_restricted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        frozen_module, "_BiomedCLIPVisionEncoder", FakeVisionEncoder
    )
    artifact, manifest = write_runtime_artifact(tmp_path)
    checkpoint = tmp_path / "biomedclip"
    checkpoint.mkdir()

    runtime = FrozenPRISMRuntime.load(
        artifact,
        manifest_path=manifest,
        image_encoder_checkpoint=str(checkpoint),
    )
    assert runtime.artifact_id == "test-prism"
