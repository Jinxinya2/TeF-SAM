from pathlib import Path

from tefsam.models.frozen_prism import FrozenPRISMRuntime


ROOT = Path(__file__).resolve().parents[1]


def test_private_builder_sources_are_absent():
    forbidden = (
        "tefsam/models/prism.py",
        "tefsam/models/prism_v3.py",
        "tefsam/models/hard_negative_loss.py",
        "scripts/build_prism.py",
        "tefsam/cli/build_prism.py",
    )
    assert not [relative for relative in forbidden if (ROOT / relative).exists()]


def test_runtime_has_no_repository_mutation_api():
    for name in ("build", "fit", "save", "update_repository"):
        assert not hasattr(FrozenPRISMRuntime, name)


def test_public_python_does_not_import_private_modules():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "tefsam").rglob("*.py")
    )
    assert "hard_negative_loss" not in source
    assert "models.prism" not in source
    assert "sinkhorn_knopp" not in source
