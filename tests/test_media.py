from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from redactor import media


def load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def parse_fixture(name: str, source_path: Path, cwd: Path) -> dict[str, Any]:
    info = media.parse_probe_result(
        load_fixture(name),
        source_path=source_path,
        file_size_bytes=123,
        sha256="0" * 64,
        ffprobe_version="ffprobe version test",
        created_at="2026-01-01T00:00:00Z",
        cwd=cwd,
    )
    return info.to_dict()


def test_sha256_file_streams_bytes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    content = (b"abc123" * 200_000) + b"tail"
    sample.write_bytes(content)

    assert media.sha256_file(sample, chunk_size=1024) == hashlib.sha256(content).hexdigest()


def test_sha256_file_cache_hits_and_invalidates_on_file_change(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"first")
    media.clear_immutable_media_cache()

    first = media.sha256_file(sample)
    second = media.sha256_file(sample)
    assert first == second
    assert media.immutable_cache_stats()["sha256_hits"] == 1

    sample.write_bytes(b"second value changes size")
    changed = media.sha256_file(sample)
    assert changed != first
    stats = media.immutable_cache_stats()
    assert stats["sha256_misses"] == 2
    assert stats["sha256_hits"] == 1


def test_parse_video_with_audio(tmp_path: Path) -> None:
    source = tmp_path / "clip with spaces.mp4"
    parsed = parse_fixture("ffprobe_video_audio.json", source, tmp_path)

    assert parsed["source_filename"] == "clip with spaces.mp4"
    assert parsed["source_path"] == "clip with spaces.mp4"
    assert parsed["container_format"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert parsed["duration_seconds"] == 300.123
    assert parsed["video_stream"]["width"] == 1920
    assert parsed["audio_stream"] == {
        "stream_index": 1,
        "codec": "aac",
        "sample_rate": 48000,
        "channels": 2,
    }


def test_parse_video_without_audio(tmp_path: Path) -> None:
    source = tmp_path / "video_only.mp4"
    parsed = parse_fixture("ffprobe_video_only.json", source, tmp_path)

    assert parsed["audio_stream"] is None
    assert "frame_count" not in parsed["video_stream"]


def test_preserves_rational_frame_rate_strings(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    parsed = parse_fixture("ffprobe_video_audio.json", source, tmp_path)

    assert parsed["video_stream"]["nominal_frame_rate"] == "30000/1001"
    assert parsed["video_stream"]["average_frame_rate"] == "30000/1001"


def test_trim_rejects_invalid_arguments(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    with pytest.raises(ValueError, match="start"):
        media.trim_video(input_path, output_path, start_seconds=-1, duration_seconds=1)

    with pytest.raises(ValueError, match="duration"):
        media.trim_video(input_path, output_path, start_seconds=0, duration_seconds=0)


def test_contact_sheet_rejects_invalid_arguments(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "sheet.jpg"

    with pytest.raises(ValueError, match="interval"):
        media.create_contact_sheet(
            input_path,
            output_path,
            interval_seconds=0,
            columns=1,
            max_frames=1,
        )

    with pytest.raises(ValueError, match="columns"):
        media.create_contact_sheet(
            input_path,
            output_path,
            interval_seconds=1,
            columns=0,
            max_frames=1,
        )

    with pytest.raises(ValueError, match="max-frames"):
        media.create_contact_sheet(
            input_path,
            output_path,
            interval_seconds=1,
            columns=1,
            max_frames=0,
        )


def test_probe_media_handles_paths_with_spaces_and_argument_lists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "clip with spaces.mp4"
    input_path.write_bytes(b"video bytes")
    fixture = load_fixture("ffprobe_video_audio.json")
    commands: list[list[str]] = []

    def fake_which(name: str) -> str:
        return name

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert isinstance(command, list)
        assert kwargs.get("shell") is not True
        if command[0] == "ffprobe" and "-version" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="ffprobe version test\n",
                stderr="",
            )
        if command[0] == "ffprobe":
            assert command[-1] == str(input_path)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(fixture), stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(media.shutil, "which", fake_which)
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    parsed = media.probe_media(input_path, cwd=tmp_path).to_dict()

    assert parsed["source_path"] == "clip with spaces.mp4"
    assert commands


def test_probe_media_cache_reuses_ffprobe_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "cached probe.mp4"
    input_path.write_bytes(b"video bytes")
    fixture = load_fixture("ffprobe_video_audio.json")
    commands: list[list[str]] = []
    media.clear_immutable_media_cache()

    def fake_which(name: str) -> str:
        return name

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffprobe" and "-version" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="ffprobe version test\n",
                stderr="",
            )
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(fixture), stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(media.shutil, "which", fake_which)
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    first = media.probe_media(input_path, cwd=tmp_path).to_dict()
    second = media.probe_media(input_path, cwd=tmp_path).to_dict()

    assert first["sha256"] == second["sha256"]
    assert [command for command in commands if "-show_streams" in command]
    assert len([command for command in commands if "-show_streams" in command]) == 1
    stats = media.immutable_cache_stats()
    assert stats["ffprobe_hits"] == 1
    assert stats["version_hits"] == 1


def test_ffmpeg_helpers_use_argument_lists_not_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input video.mp4"
    output_clip = tmp_path / "output clip.mp4"
    output_sheet = tmp_path / "sheet image.jpg"
    fixture = load_fixture("ffprobe_video_audio.json")
    commands: list[list[str]] = []

    def fake_which(name: str) -> str:
        return name

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert isinstance(command, list)
        assert kwargs.get("shell") is not True
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(fixture), stderr="")
        if command[0] == "ffmpeg":
            assert str(input_path) in command
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(media.shutil, "which", fake_which)
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    media.trim_video(input_path, output_clip, start_seconds=0, duration_seconds=1)
    media.create_contact_sheet(
        input_path,
        output_sheet,
        interval_seconds=2,
        columns=3,
        max_frames=6,
    )

    assert any(command[0] == "ffmpeg" for command in commands)
