import os
import json
import requests

FREECONVERT_API_URL = "https://api.freeconvert.com/v1/process/jobs"


def infer_input_format(filename: str) -> str:
    """Infer a FreeConvert input format from the file extension."""
    if not filename:
        return ""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext


def build_freeconvert_payload(input_format: str, output_format: str) -> dict:
    """Build the request payload for a simple audio conversion job."""
    return {
        "tasks": {
            "import": {
                "operation": "import/upload"
            },
            "convert": {
                "operation": "convert",
                "input": "import",
                "input_format": input_format,
                "output_format": output_format,
                "options": {
                    "audio_codec": "auto",
                    "audio_filter_volume": 100,
                    "audio_filter_fade_in": False,
                    "audio_filter_fade_out": False,
                    "audio_filter_reverse": False,
                    "cut_start": "00:00:00.00",
                    "cut_end": "00:00:00.00"
                }
            },
            "export-url": {
                "operation": "export/url",
                "input": ["convert"]
            }
        }
    }


def convert_audio_with_freeconvert(audio_bytes: bytes, filename: str, output_format: str,
                                   access_token: str):
    """Submit a conversion job to FreeConvert and return the response payload."""
    if not access_token:
        raise RuntimeError("FreeConvert access token is missing.")

    input_format = infer_input_format(filename)
    if not input_format:
        raise RuntimeError("Unable to infer the source audio format from the file name.")

    payload = build_freeconvert_payload(input_format, output_format)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    try:
        response = requests.post(
            FREECONVERT_API_URL,
            data=json.dumps(payload),
            headers=headers,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"FreeConvert request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"FreeConvert returned HTTP {response.status_code}: {response.text[:300]}"
        )

    return response.json()
