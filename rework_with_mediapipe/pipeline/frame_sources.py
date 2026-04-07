from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from pipeline.types import EpisodeRef


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


class LegacyRawFrameSource(FrameSource):
    def read_frame(self, frame_idx: int) -> Image.Image:
        assert self.episode.crop_dir is not None
        frame_path = Path(self.episode.crop_dir) / "extracted_images" / f"{frame_idx:06d}.jpg"
        return Image.open(frame_path).convert("RGB")


class FactoryTarFrameSource(FrameSource):
    def read_frame(self, frame_idx: int) -> Image.Image:
        assert self.episode.frame_names is not None
        assert self.episode.shard_path is not None
        if self.episode.frame_offsets is not None:
            offset, size = self.episode.frame_offsets[frame_idx]
            with open(self.episode.shard_path, "rb") as f:
                f.seek(offset)
                payload = f.read(size)
        else:
            with tarfile.open(self.episode.shard_path, "r") as tar:
                member = tar.getmember(self.episode.frame_names[frame_idx])
                payload = tar.extractfile(member).read()
        return Image.open(io.BytesIO(payload)).convert("RGB")


def make_frame_source(episode: EpisodeRef) -> FrameSource:
    if episode.source_type == "legacy_raw":
        return LegacyRawFrameSource(episode)
    if episode.source_type == "factory_tar":
        return FactoryTarFrameSource(episode)
    raise ValueError(f"Unsupported source type: {episode.source_type}")
