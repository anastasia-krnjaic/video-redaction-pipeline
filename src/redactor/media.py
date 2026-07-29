from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from redactor.models import AudioStreamInfo, MediaInfo, VideoStreamInfo

SCHEMA_VERSION = "1.0"
HASH_CHUNK_SIZE = 1024 * 1024
CONTACT_SHEET_TILE_WIDTH = 320


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} was not found on PATH")
    return path


def version_line(name: str) -> str:
    executable = executable_path(name)
    result = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()[0] if result.stdout else f"{name} version unavailable"


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def portable_source_path(path: Path, cwd: Path | None = None) -> str:
    resolved_path = path.resolve()
    resolved_cwd = (cwd or Path.cwd()).resolve()
    try:
        return resolved_path.relative_to(resolved_cwd).as_posix()
    except ValueError:
        return path.name


def probe_media(path: Path, cwd: Path | None = None) -> MediaInfo:
    probe = _run_ffprobe(path)
    return parse_probe_result(
        probe,
        source_path=path,
        file_size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        ffprobe_version=version_line("ffprobe"),
        created_at=utc_timestamp(),
        cwd=cwd,
    )


def parse_probe_result(
    probe: dict[str, Any],
    *,
    source_path: Path,
    file_size_bytes: int,
    sha256: str,
    ffprobe_version: str,
    created_at: str,
    cwd: Path | None = None,
) -> MediaInfo:
    streams = probe.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError("no video stream found")

    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    format_info = probe.get("format") or {}

    return MediaInfo(
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        source_filename=source_path.name,
        source_path=portable_source_path(source_path, cwd),
        file_size_bytes=file_size_bytes,
        sha256=sha256,
        container_format=_optional_string(format_info.get("format_name")),
        duration_seconds=_optional_float(format_info.get("duration")),
        video_stream=_parse_video_stream(video_stream),
        audio_stream=_parse_audio_stream(audio_stream) if audio_stream else None,
        ffprobe_version=ffprobe_version,
    )


def write_media_info(info: MediaInfo, output_path: Path) -> None:
    output_path.write_text(json.dumps(info.to_dict(), indent=2) + "\n", encoding="utf-8")


def trim_video(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool = False,
) -> float | None:
    if start_seconds < 0:
        raise ValueError("start must be greater than or equal to 0")
    if duration_seconds <= 0:
        raise ValueError("duration must be greater than 0")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    has_audio = _has_audio_stream(input_path)
    command = [
        executable_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        _format_seconds(start_seconds),
        "-i",
        str(input_path),
        "-t",
        _format_seconds(duration_seconds),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    subprocess.run(command, capture_output=True, text=True, check=True)
    return _probe_duration(output_path)


def create_contact_sheet(
    input_path: Path,
    output_path: Path,
    *,
    interval_seconds: float,
    columns: int,
    max_frames: int,
    overwrite: bool = False,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval-seconds must be greater than 0")
    if columns <= 0:
        raise ValueError("columns must be greater than 0")
    if max_frames <= 0:
        raise ValueError("max-frames must be greater than 0")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    rows = math.ceil(max_frames / columns)
    filtergraph = (
        f"fps=1/{_format_seconds(interval_seconds)},"
        f"scale={CONTACT_SHEET_TILE_WIDTH}:-1:flags=lanczos,"
        f"tile={columns}x{rows}:nb_frames={max_frames}:padding=4:margin=4:color=white"
    )
    command = [
        executable_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        filtergraph,
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def _run_ffprobe(path: Path) -> dict[str, Any]:
    command = [
        executable_path("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _has_audio_stream(path: Path) -> bool:
    probe = _run_ffprobe(path)
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _probe_duration(path: Path) -> float | None:
    probe = _run_ffprobe(path)
    format_info = probe.get("format") or {}
    return _optional_float(format_info.get("duration"))


def _parse_video_stream(stream: dict[str, Any]) -> VideoStreamInfo:
    return VideoStreamInfo(
        index=_required_int(stream.get("index"), "video stream index"),
        codec=_optional_string(stream.get("codec_name")),
        width=_required_int(stream.get("width"), "video width"),
        height=_required_int(stream.get("height"), "video height"),
        pixel_format=_optional_string(stream.get("pix_fmt")),
        nominal_frame_rate=_optional_string(stream.get("r_frame_rate")),
        average_frame_rate=_optional_string(stream.get("avg_frame_rate")),
        time_base=_optional_string(stream.get("time_base")),
        frame_count=_optional_string(stream.get("nb_frames")),
    )


def _parse_audio_stream(stream: dict[str, Any]) -> AudioStreamInfo:
    return AudioStreamInfo(
        index=_required_int(stream.get("index"), "audio stream index"),
        codec=_optional_string(stream.get("codec_name")),
        sample_rate=_optional_int(stream.get("sample_rate")),
        channels=_optional_int(stream.get("channels")),
    )


def _required_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _format_seconds(value: float) -> str:
    return f"{value:g}"
