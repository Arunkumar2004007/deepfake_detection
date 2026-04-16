"""
models/demux.py — Audio/Video stream demultiplexer using ffmpeg
Splits a .webm/.mp4 recording into separate video and audio files.
"""
import os
import subprocess
import tempfile


def demux(input_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Split input_path into video-only and audio-only files.
    Returns: (video_path, audio_path) — either may be None on failure.

    Requirements: ffmpeg must be installed and available in PATH.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    video_path = os.path.join(output_dir, f"{base}_video.mp4")
    audio_path = os.path.join(output_dir, f"{base}_audio.wav")

    # Extract video stream (no audio)
    video_ok = _run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-an", "-vcodec", "copy",
        video_path
    ])

    # Extract audio stream (no video), resample to 16 kHz mono WAV
    audio_ok = _run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ])

    return (video_path if video_ok else None,
            audio_path if audio_ok else None)


def mux(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Combine a video-only and audio-only file into a single output file.
    Returns True on success.
    """
    return _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        output_path
    ])


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio track from a video file to WAV format."""
    return _run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ])


def _run_ffmpeg(cmd: list) -> bool:
    """Run an ffmpeg command, return True on success."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        if result.returncode != 0:
            print(f"[ffmpeg] Error: {result.stderr.decode()[:300]}")
            return False
        return True
    except FileNotFoundError:
        print("[ffmpeg] ffmpeg not found in PATH. Install ffmpeg to enable demux.")
        return False
    except subprocess.TimeoutExpired:
        print("[ffmpeg] Timeout expired.")
        return False
    except Exception as e:
        print(f"[ffmpeg] Exception: {e}")
        return False
