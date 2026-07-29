from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int
    codec: str | None
    width: int
    height: int
    pixel_format: str | None
    nominal_frame_rate: str | None
    average_frame_rate: str | None
    time_base: str | None
    frame_count: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("video width must be positive")
        if self.height <= 0:
            raise ValueError("video height must be positive")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "stream_index": self.index,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "nominal_frame_rate": self.nominal_frame_rate,
            "average_frame_rate": self.average_frame_rate,
            "time_base": self.time_base,
        }
        if self.frame_count is not None:
            data["frame_count"] = self.frame_count
        return data


@dataclass(frozen=True)
class AudioStreamInfo:
    index: int
    codec: str | None
    sample_rate: int | None
    channels: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_index": self.index,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class MediaInfo:
    schema_version: str
    created_at: str
    source_filename: str
    source_path: str
    file_size_bytes: int
    sha256: str
    container_format: str | None
    duration_seconds: float | None
    video_stream: VideoStreamInfo
    audio_stream: AudioStreamInfo | None
    ffprobe_version: str

    def __post_init__(self) -> None:
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source_filename": self.source_filename,
            "source_path": self.source_path,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "container_format": self.container_format,
            "duration_seconds": self.duration_seconds,
            "video_stream": self.video_stream.to_dict(),
            "audio_stream": self.audio_stream.to_dict() if self.audio_stream else None,
            "ffprobe_version": self.ffprobe_version,
        }
