import os
import subprocess

SUPPORTED_AUDIO_EXTENSIONS = ["wav", "mp3", "ogg", "m4a", "flac"]


def get_supported_audio_extensions() -> list[str]:
    """Return the audio extensions the app accepts for upload."""
    return list(SUPPORTED_AUDIO_EXTENSIONS)


def is_supported_audio_file(filename: str) -> bool:
    """Return True when the filename looks like a supported audio upload."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in set(SUPPORTED_AUDIO_EXTENSIONS)


def should_convert_to_wav(filename: str) -> bool:
    """Return True when the uploaded file needs WAV normalization before Azure."""
    if not filename:
        return True
    ext = os.path.splitext(filename)[1].lower()
    return ext not in {".wav", ".wave"}


def prepare_audio_for_transcription(input_path: str, output_dir: str | None = None) -> str:
    """Prepare an audio file for Azure transcription.

    For WAV inputs we reuse the file directly to avoid an unnecessary ffmpeg pass.
    For non-WAV audio, we convert to a mono 16kHz WAV once so Azure can consume it.
    """
    import imageio_ffmpeg

    ext = os.path.splitext(input_path)[1].lower()
    if ext in {".wav", ".wave"}:
        return input_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = os.path.join(output_dir or os.path.dirname(input_path), "prepared.wav")
    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not prepare audio for transcription: {result.stderr[-300:]}")
    return output_path
