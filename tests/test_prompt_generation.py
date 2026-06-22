import types
from types import SimpleNamespace

import torch
import torch.nn as nn

from tefsam.models.sam_adapter import MedicalSAMAdapter


def prompt_sampler(
    *, point_topk=1, threshold=0.9, correction_points_per_class=2
):
    adapter = MedicalSAMAdapter.__new__(MedicalSAMAdapter)
    nn.Module.__init__(adapter)
    adapter.input_image_size = (8, 8)
    adapter.point_topk = point_topk
    adapter.point_threshold = threshold
    adapter.correction_points_per_class = correction_points_per_class
    return adapter


def test_initial_positive_point_comes_from_coarse_mask_peak():
    adapter = prompt_sampler()
    coarse_logits = torch.full((1, 1, 4, 4), -10.0)
    coarse_logits[0, 0, 1, 2] = 10.0

    coords, labels = adapter._initial_points_from_coarse_mask(coarse_logits)

    assert torch.allclose(coords[0, 0], torch.tensor([4.5, 2.5]))
    assert labels.tolist() == [[1]]


def test_points_use_loaded_sam_prompt_encoder_coordinate_system():
    adapter = prompt_sampler()
    adapter.prompt_encoder = SimpleNamespace(input_image_size=(1024, 1024))
    coarse_logits = torch.full((1, 1, 4, 4), -10.0)
    coarse_logits[0, 0, 1, 2] = 10.0

    coords, _ = adapter._initial_points_from_coarse_mask(coarse_logits)

    assert torch.allclose(coords[0, 0], torch.tensor([639.5, 383.5]))


def test_low_confidence_fallback_uses_distinct_top_scoring_candidates():
    adapter = prompt_sampler(point_topk=2, threshold=0.999)
    coarse_logits = torch.tensor([[[[2.0, 1.0], [-2.0, -3.0]]]])

    coords, labels = adapter._initial_points_from_coarse_mask(coarse_logits)

    assert coords.unique(dim=1).size(1) == 2
    assert {tuple(point.tolist()) for point in coords[0]} == {
        (1.5, 1.5),
        (5.5, 1.5),
    }
    assert labels.tolist() == [[1, 1]]


def test_correction_points_alternate_false_negative_and_false_positive():
    adapter = prompt_sampler(correction_points_per_class=2)
    prediction = torch.full((1, 1, 8, 8), -10.0)
    prediction[0, 0, 7, 7] = 10.0
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 0, 0] = 1.0

    coords, labels = adapter._correction_points(prediction, target)

    assert labels.tolist() == [[1, 0, -1, -1]]
    assert torch.allclose(coords[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(coords[0, 1], torch.tensor([7.0, 7.0]))


def test_ground_truth_corrections_are_training_only():
    adapter = MedicalSAMAdapter(
        image_dim=8,
        text_dim=8,
        prompt_dim=8,
        image_embedding_size=2,
        input_image_size=(8, 8),
        point_topk=1,
        point_threshold=0.5,
        correction_points_per_class=1,
    )
    decode_calls = []
    original_decode = adapter._decode_mask

    def decode_spy(self, *args, **kwargs):
        decode_calls.append(1)
        return original_decode(*args, **kwargs)

    def fixed_corrections(self, prediction_logits, target):
        del prediction_logits, target
        return (
            torch.zeros(1, 2, 2),
            torch.tensor([[1, 0]], dtype=torch.long),
        )

    adapter._decode_mask = types.MethodType(decode_spy, adapter)
    adapter._correction_points = types.MethodType(fixed_corrections, adapter)
    image_tokens = torch.randn(1, 4, 8)
    text_tokens = torch.randn(1, 2, 8)
    target = torch.zeros(1, 1, 8, 8)

    adapter.train()
    train_logits, _ = adapter(
        image_tokens, text_tokens, target_size=(8, 8), target=target
    )
    assert train_logits.shape == target.shape
    assert len(decode_calls) == 2

    decode_calls.clear()
    adapter.eval()
    with torch.no_grad():
        eval_logits_without_lesion, _ = adapter(
            image_tokens, text_tokens, target_size=(8, 8), target=target
        )
        eval_logits_with_lesion, _ = adapter(
            image_tokens,
            text_tokens,
            target_size=(8, 8),
            target=torch.ones_like(target),
        )
    assert eval_logits_without_lesion.shape == target.shape
    assert torch.allclose(eval_logits_without_lesion, eval_logits_with_lesion)
    assert len(decode_calls) == 2
