from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import Proposal


@dataclass
class ChainState:
    chain_index: int
    proposals: List[Proposal] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def last(self) -> Proposal:
        return self.proposals[-1]

    def add(self, proposal: Proposal) -> None:
        self.proposals.append(proposal)
        self.missed_frames = 0


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _wrist_distance_px(a: Proposal, b: Proposal) -> float:
    pa = np.array(a.landmarks_2d[0][:2], dtype=np.float32)
    pb = np.array(b.landmarks_2d[0][:2], dtype=np.float32)
    return float(np.linalg.norm(pa - pb))


def _normal_angle_deg(a: Proposal, b: Proposal) -> float:
    va = np.array(a.palm_normal, dtype=np.float32)
    vb = np.array(b.palm_normal, dtype=np.float32)
    na = max(float(np.linalg.norm(va)), 1e-6)
    nb = max(float(np.linalg.norm(vb)), 1e-6)
    cosine = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _association_score(prev: Proposal, curr: Proposal, image_diag: float) -> float:
    iou = _bbox_iou(prev.bbox_xyxy, curr.bbox_xyxy)
    wrist = _wrist_distance_px(prev, curr)
    wrist_term = max(0.0, 1.0 - wrist / max(image_diag * 0.12, 1.0))
    hand_term = 1.0 - min(1.0, abs(prev.handedness_score - curr.handedness_score))
    normal_angle = _normal_angle_deg(prev, curr)
    normal_term = max(0.0, 1.0 - normal_angle / 80.0)
    return 0.35 * iou + 0.35 * wrist_term + 0.10 * hand_term + 0.20 * normal_term


def build_candidate_chains(
    per_frame_proposals: Dict[int, List[Proposal]],
    *,
    image_size: Tuple[int, int],
    max_gap: int,
    min_chain_frames: int,
) -> List[Dict]:
    image_diag = float(math.hypot(*image_size))
    active: List[ChainState] = []
    finished: List[ChainState] = []
    next_chain_index = 0

    for frame_idx in sorted(per_frame_proposals.keys()):
        proposals = per_frame_proposals[frame_idx]
        used = set()
        for chain in list(active):
            best_idx: Optional[int] = None
            best_score = 0.0
            for idx, proposal in enumerate(proposals):
                if idx in used:
                    continue
                score = _association_score(chain.last, proposal, image_diag)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= 0.25:
                chain.add(proposals[best_idx])
                used.add(best_idx)
            else:
                chain.missed_frames += 1
            if chain.missed_frames > max_gap:
                active.remove(chain)
                finished.append(chain)
        for idx, proposal in enumerate(proposals):
            if idx in used:
                continue
            active.append(ChainState(chain_index=next_chain_index, proposals=[proposal]))
            next_chain_index += 1

    finished.extend(active)
    chains: List[Dict] = []
    for chain in finished:
        if len(chain.proposals) < min_chain_frames:
            continue
        frames = [proposal.frame_idx for proposal in chain.proposals]
        avg_score = float(sum(item.score for item in chain.proposals) / len(chain.proposals))
        coverage_score = len(frames) + avg_score
        role_hint = "unknown"
        mean_handedness = float(sum(item.handedness_score for item in chain.proposals) / len(chain.proposals))
        if mean_handedness >= 0.5:
            role_hint = "likely_right"
        elif mean_handedness <= -0.5:
            role_hint = "likely_left"
        preview_idx = frames[len(frames) // 2]
        chains.append(
            {
                "chain_index": chain.chain_index,
                "role_hint": role_hint,
                "score": coverage_score,
                "start_frame": min(frames),
                "end_frame": max(frames),
                "num_frames": len(frames),
                "preview_frame": preview_idx,
                "frames": frames,
                "proposals": [proposal.to_dict() for proposal in chain.proposals],
            }
        )
    chains.sort(key=lambda item: item["score"], reverse=True)
    for new_idx, chain in enumerate(chains):
        chain["chain_index"] = new_idx
    return chains


def build_merge_questions(chains: Sequence[Dict], max_questions: int = 2) -> List[Dict]:
    questions: List[Dict] = []
    for i, left in enumerate(chains):
        for right in chains[i + 1 :]:
            gap = right["start_frame"] - left["end_frame"]
            if gap < 1 or gap > 15:
                continue
            if left["role_hint"] != right["role_hint"]:
                continue
            questions.append(
                {
                    "question_id": f"merge_{left['chain_index']}_{right['chain_index']}",
                    "chain_a": left["chain_index"],
                    "chain_b": right["chain_index"],
                    "prompt": f"候选 {left['chain_index']} 和 {right['chain_index']} 是否是同一只手离开后又回来？",
                }
            )
            if len(questions) >= max_questions:
                return questions
    return questions
