from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRef:
    episode_id: str
    episode_name: str
    dataset_name: str
    source_type: str
    num_frames: int
    shard_path: Optional[str] = None
    seq_folder: Optional[str] = None
    frame_names: Optional[List[str]] = None
    frame_offsets: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClipRef:
    episode: EpisodeRef
    clip_start: int
    clip_end: int
    dirty_reason: str
    bad_frames: List[int] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return max(0, self.clip_end - self.clip_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode.to_dict(),
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "dirty_reason": self.dirty_reason,
            "bad_frames": list(self.bad_frames),
        }


@dataclass
class Proposal:
    frame_idx: int
    det_idx: int
    bbox_xyxy: List[float]
    score: float
    handedness_score: float
    handedness_label: str
    side_rank: int
    landmarks_2d: List[List[float]]
    landmarks_3d_rel: List[List[float]]
    roi_rotation_2d: float
    palm_normal: List[float]
    wrist_frame: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
