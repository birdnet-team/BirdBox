# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Models with a bird class: boxes from the bird score, names from the species."""

import pytest

from inference import detect_birds as db
from inference.utils import bird_labels as bl
from conftest import make_det

BIRD = 3  # three species, then bird


def head(*anchors):
    """Head output (1, 4 + 4 classes, anchors) from (cx, cy, w, h, s0, s1, s2, bird) rows."""
    torch = pytest.importorskip("torch")  # CI runs the unit tests without PyTorch
    pytest.importorskip("torchvision")
    return torch.tensor(anchors, dtype=torch.float32).T.unsqueeze(0)


def test_bird_class_id():
    assert bl.bird_class_id({0: 'a', 1: 'b', 2: 'bird'}) == 2
    assert bl.bird_class_id(['a', 'Bird']) == 1
    assert bl.bird_class_id({0: 'a', 1: 'b'}) is None
    assert bl.bird_class_id({}) is None


def test_nms_on_bird_score_with_pooled_species():
    pred = head(
        (50, 50, 20, 20, 0.1, 0.6, 0.0, 0.9),   # kept: strongest bird score
        (52, 50, 20, 20, 0.7, 0.0, 0.0, 0.8),   # same call: suppressed, its species pooled
        (200, 50, 20, 20, 0.0, 0.0, 0.3, 0.5),  # another call
        (120, 50, 20, 20, 0.9, 0.0, 0.0, 0.1),  # species alone, bird below conf: no box
    )
    out = bl.bird_label_nms(pred, conf=0.2, iou=0.5, bird_id=BIRD, pool=0.5, top_k=2)[0]
    assert out.shape == (2, 5 + 4)
    assert out[:, 4].tolist() == pytest.approx([0.9, 0.5])
    assert out[0, 5:7].tolist() == [0, 1]                    # pooled: species 0 (0.7) over 1 (0.6)
    assert out[0, 7:9].tolist() == pytest.approx([0.7, 0.6])
    assert out[1, 5].item() == 2

    own = bl.bird_label_nms(pred, conf=0.2, iou=0.5, bird_id=BIRD, pool=0, top_k=1)[0]
    assert own[0, 5].item() == 1                             # own anchor only


def test_label_threshold():
    names = {0: 'amerob', 1: 'houfin', 2: 'blujay', BIRD: 'bird'}
    named = bl.label_species([1, 0], [0.4, 0.1], names, BIRD, label_conf=0.15)
    assert (named['species'], named['species_id'], named['species_confidence']) == ('houfin', 1, 0.4)
    assert [a['species'] for a in named['alternatives']] == ['houfin', 'amerob']
    unnamed = bl.label_species([1], [0.1], names, BIRD, label_conf=0.15)
    assert (unnamed['species'], unnamed['species_id']) == ('bird', BIRD)


def test_merge_is_species_blind():
    def raw(t0, t1, bird, alts, clip):
        named = bl._named([{'species': s, 'species_id': i, 'confidence': c} for s, i, c in alts], BIRD, 0.15)
        return make_det(t0, t1, 1000, 3000, confidence=bird, clip_start=clip, **named)

    # One call seen by two windows, named differently in each.
    dets = [raw(1.6, 2.0, 0.8, [('houfin', 1, 0.3)], 0.0),
            raw(1.62, 2.02, 0.7, [('amerob', 0, 0.5), ('houfin', 1, 0.2)], 1.5)]
    merged = db.merge_detections(dets, 'fusion', 0.1, phrase_gap=0, file_end=4.5)
    assert len(merged) == 1
    assert merged[0]['species'] == 'amerob'
    assert [a['confidence'] for a in merged[0]['alternatives']] == [0.5, 0.3]

    weak = [raw(1.6, 2.0, 0.8, [('houfin', 1, 0.1)], 0.0)]
    assert db.merge_detections(weak, 'reconstruct', 0.1)[0]['species'] == 'bird'
