# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Boxes from a bird class, labels from the species scores of the same box.

A model whose last class is ``bird`` (Birds-With-Labels with the bird class) learns
that class from every labelled box, whatever its species. It finds calls like
Just-Bird. Its species classes learn from their own boxes only, so they fire on
fewer calls, and Ultralytics' per-class NMS gives one call several boxes with
different species.

``bird_label_nms`` replaces that NMS for such a model:

1. Each anchor's box is scored by its bird score. Boxes above the confidence go
   through class-agnostic NMS at the given IoU.
2. Each kept box takes the species scores of the anchors around it: per species,
   the maximum over the boxes above the confidence that overlap it at IoU >= ``pool``
   (the box's own anchor is one of them). ``pool`` 0 uses the box's own anchor only.
3. The output per box is its coordinates, the bird score, and the top ``top_k``
   species with their scores.

``label_species`` then names the box after its top species when that score reaches
the label threshold, and ``bird`` otherwise. ``merge_and_label`` merges such boxes
species-blind (fusion or song merge) and names each merged box the same way, from the
raw boxes inside it.
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

BIRD_CLASS_NAME = "bird"
# Species scores pooled over the anchors that overlap a kept box at this IoU.
DEFAULT_POOL_IOU = 0.5
# A box is named after its top species when that species' score reaches this.
# Joined val (Birds-With-Labels, bird class), best macro F1 at 0.15 / 0.25 / 0.35:
# presence per recording 0.591 / 0.627 / 0.617, bout-aware inside 0.462 / 0.469 /
# 0.447, either way 0.471 / 0.478 / 0.458, per 3 s 0.516 / 0.510 / 0.475.
DEFAULT_LABEL_CONF = 0.25
TOP_K = 3


def bird_class_id(names) -> Optional[int]:
    """The id of the bird class when it is the model's last class, else None."""
    if not names:
        return None
    if isinstance(names, dict):
        last = max(int(k) for k in names)
        name = names.get(last, names.get(str(last)))
    else:
        last, name = len(names) - 1, names[-1]
    return last if str(name).strip().lower() == BIRD_CLASS_NAME else None


def bird_label_nms(
    prediction,
    conf: float,
    iou: float,
    bird_id: int,
    pool: float = DEFAULT_POOL_IOU,
    top_k: int = TOP_K,
    max_det: int = 300,
):
    """Class-agnostic NMS on the bird score, with pooled species scores per box.

    Args:
        prediction: Head output ``(batch, 4 + classes, anchors)`` with boxes as
            centre x, centre y, width, height in input pixels, or a tuple whose
            first item is that tensor.
        conf: Bird score threshold.
        iou: NMS IoU threshold.
        bird_id: Index of the bird class; the species are the classes before it.
        pool: IoU at which a kept box takes the species scores of other boxes.
        top_k: Species kept per box.
        max_det: Boxes kept per image.

    Returns:
        One tensor per image, ``(boxes, 5 + 2 * top_k)``: x1, y1, x2, y2, bird
        score, then the top species ids and their scores.
    """
    import torch
    import torchvision

    x = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    k = min(top_k, bird_id)
    out = []
    for p in x:
        p = p.transpose(0, 1)
        p = p[p[:, 4 + bird_id] > conf]
        centre, half = p[:, :2], p[:, 2:4] / 2
        boxes = torch.cat((centre - half, centre + half), 1)
        bird = p[:, 4 + bird_id]
        species = p[:, 4:4 + bird_id]
        keep = torchvision.ops.nms(boxes, bird, iou)[:max_det]
        kept = boxes[keep]
        if pool > 0 and len(keep):
            rows, cols = (torchvision.ops.box_iou(kept, boxes) >= pool).nonzero(as_tuple=True)
            pooled = torch.zeros(len(keep), bird_id, dtype=p.dtype, device=p.device)
            pooled.scatter_reduce_(0, rows[:, None].expand(-1, bird_id), species[cols], "amax")
        else:
            pooled = species[keep]
        scores, ids = pooled.topk(k, dim=1)
        out.append(torch.cat((kept, bird[keep, None], ids.to(p.dtype), scores), 1))
    return out


def make_predictor(bird_id: int, pool: float = DEFAULT_POOL_IOU, top_k: int = TOP_K):
    """An Ultralytics DetectionPredictor that runs ``bird_label_nms``.

    Each Results keeps the boxes with the bird score as confidence and the top
    species as class, and gets ``species_topk``: per box, the top species ids and
    their scores (``(boxes, 2 * top_k)``).
    """
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops

    class BirdLabelPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            preds = bird_label_nms(
                preds, self.args.conf, self.args.iou, bird_id, pool, top_k, self.args.max_det
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            results = []
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
                k = (pred.shape[1] - 5) // 2
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                boxes = pred[:, :6]  # x1 y1 x2 y2, bird score, top species
                result = Results(orig_img, path=img_path, names=self.model.names, boxes=boxes)
                result.species_topk = pred[:, 5:5 + 2 * k]
                results.append(result)
            return results

    return BirdLabelPredictor


def label_species(
    ids: Sequence[int],
    scores: Sequence[float],
    id_to_ebird: Dict[int, str],
    bird_id: int,
    label_conf: float = DEFAULT_LABEL_CONF,
) -> Dict:
    """Name a bird box from its top species.

    Returns the detection fields: ``species`` and ``species_id`` (the top species,
    or ``bird`` and ``bird_id`` below ``label_conf``), ``species_confidence`` (the top species' score) and
    ``alternatives`` (every top species with its score, best first).
    """
    alternatives: List[Dict] = [
        {"species": id_to_ebird.get(int(i), f"unknown_{int(i)}"), "species_id": int(i), "confidence": float(s)}
        for i, s in zip(ids, scores)
    ]
    return _named(alternatives, bird_id, label_conf)


def _named(alternatives: List[Dict], bird_id: int, label_conf: float) -> Dict:
    top = alternatives[0] if alternatives else None
    named = top is not None and top["confidence"] >= label_conf
    return {
        "species": top["species"] if named else BIRD_CLASS_NAME,
        "species_id": top["species_id"] if named else bird_id,
        "bird_id": bird_id,
        "species_confidence": top["confidence"] if top else 0.0,
        "alternatives": alternatives,
    }


def is_bird_labelled(detections: Sequence[Dict]) -> bool:
    """True when the detections come from a bird-class model (they carry alternatives)."""
    return any("alternatives" in d for d in detections)


def merge_and_label(
    detections: List[Dict],
    merge: Callable[[List[Dict]], List[Dict]],
    label_conf: float = DEFAULT_LABEL_CONF,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Merge bird boxes species-blind, then name each merged box.

    ``merge`` runs on the detections with one shared class, as for Just-Bird, so one
    call never ends up in two boxes with different names. Each merged box then takes,
    per species, the highest score among the raw boxes whose centre lies inside it,
    and is named after the best one (``bird`` below ``label_conf``).
    """
    if not detections:
        return []
    bird_id = next(int(d["bird_id"]) for d in detections if "bird_id" in d)
    blind = [
        {**{k: v for k, v in d.items() if k != "alternatives"}, "species": BIRD_CLASS_NAME, "species_id": bird_id}
        for d in detections
    ]
    merged = merge(blind)
    by_file: Dict[str, List[Dict]] = {}
    for d in detections:
        by_file.setdefault(d.get("filename", "__single_file__"), []).append(d)
    centres = {
        name: (
            np.array([(d["time_start"] + d["time_end"]) / 2 for d in dets]),
            np.array([(d["freq_low_hz"] + d["freq_high_hz"]) / 2 for d in dets]),
        )
        for name, dets in by_file.items()
    }
    for song in merged:
        name = song.get("filename", "__single_file__")
        dets = by_file.get(name, [])
        t, f = centres.get(name, (np.zeros(0), np.zeros(0)))
        inside = np.nonzero(
            (t >= song["time_start"]) & (t <= song["time_end"])
            & (f >= song["freq_low_hz"]) & (f <= song["freq_high_hz"])
        )[0]
        best: Dict[int, Dict] = {}
        for i in inside:
            for alt in dets[i].get("alternatives", ()):
                if alt["species_id"] not in best or alt["confidence"] > best[alt["species_id"]]["confidence"]:
                    best[alt["species_id"]] = alt
        ranked = sorted(best.values(), key=lambda a: -a["confidence"])[:top_k]
        song.update(_named([dict(a) for a in ranked], bird_id, label_conf))
    return merged
