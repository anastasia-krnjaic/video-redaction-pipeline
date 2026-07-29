from __future__ import annotations

import pytest

from redactor.models import MediaInfo, VideoStreamInfo


def valid_video_stream() -> VideoStreamInfo:
    return VideoStreamInfo(
        index=0,
        codec="h264",
        width=1920,
        height=1080,
        pixel_format="yuv420p",
        nominal_frame_rate="30000/1001",
        average_frame_rate="30000/1001",
        time_base="1/30000",
    )


def test_rejects_invalid_width_and_height() -> None:
    with pytest.raises(ValueError, match="width"):
        VideoStreamInfo(
            index=0,
            codec="h264",
            width=0,
            height=1080,
            pixel_format=None,
            nominal_frame_rate=None,
            average_frame_rate=None,
            time_base=None,
        )

    with pytest.raises(ValueError, match="height"):
        VideoStreamInfo(
            index=0,
            codec="h264",
            width=1920,
            height=-1,
            pixel_format=None,
            nominal_frame_rate=None,
            average_frame_rate=None,
            time_base=None,
        )


def test_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        MediaInfo(
            schema_version="1.0",
            created_at="2026-01-01T00:00:00Z",
            source_filename="sample.mp4",
            source_path="sample.mp4",
            file_size_bytes=123,
            sha256="0" * 64,
            container_format="mov,mp4,m4a,3gp,3g2,mj2",
            duration_seconds=-0.1,
            video_stream=valid_video_stream(),
            audio_stream=None,
            ffprobe_version="ffprobe version test",
        )
