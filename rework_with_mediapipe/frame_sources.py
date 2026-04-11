from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .types import EpisodeRef


@dataclass
class FramePacket:
    frame_idx: int
    image: Image.Image


class FrameSource:
    def __init__(self, episode: EpisodeRef):
        self.episode = episode

    def read_frame(self, frame_idx: int) -> Image.Image:
        raise NotImplementedError

    def iter_frames(self, start: int, end: int) -> Iterable[FramePacket]:
        for frame_idx in range(start, end):
            yield FramePacket(frame_idx=frame_idx, image=self.read_frame(frame_idx))


class FactoryTarFrameSource(FrameSource):
    def read_frame(self, frame_idx: int) -> Image.Image:
        assert self.episode.frame_names is not None
        assert self.episode.shard_path is not None
        if self.episode.frame_offsets is not None:
            offset, size = self.episode.frame_offsets[frame_idx]
            with Path(self.episode.shard_path).open("rb") as f:
                f.seek(offset)
                payload = f.read(size)
        else:
            with tarfile.open(self.episode.shard_path, "r") as tar:
                member = tar.getmember(self.episode.frame_names[frame_idx])
                payload = tar.extractfile(member).read()
        return Image.open(io.BytesIO(payload)).convert("RGB")


def make_frame_source(episode: EpisodeRef) -> FrameSource:
    if episode.source_type != "factory_tar":
        raise ValueError(f"Unsupported source type: {episode.source_type}")
    return FactoryTarFrameSource(episode)
