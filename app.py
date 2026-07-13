"""
Lecture Audio Transcriber — a simple lecture audio transcription web app.

Two ways to transcribe with Azure Speech-to-Text:
  - Upload a pre-recorded lecture (WAV/MP3) -> Azure Fast Transcription.
  - Live microphone, captured in the browser (works on any device) and streamed
    to Azure for real-time transcription.
"""

import json
import os
import subprocess
import tempfile
import threading
import time

import requests
import streamlit as st
try:
    import imageio_ffmpeg
    _IMAGEIO_FFMPEG_ERROR = None
except Exception as err:  # pragma: no cover - optional system dependency
    imageio_ffmpeg = None
    _IMAGEIO_FFMPEG_ERROR = err
from dotenv import load_dotenv

try:
    import av
    _AV_IMPORT_ERROR = None
except Exception as err:  # pragma: no cover - cloud-only dependency issue
    av = None
    _AV_IMPORT_ERROR = err

try:
    import azure.cognitiveservices.speech as speechsdk
    _SPEECH_IMPORT_ERROR = None
except Exception as err:  # pragma: no cover - cloud-only dependency issue
    speechsdk = None
    _SPEECH_IMPORT_ERROR = err

try:
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer
    _WEBRTC_IMPORT_ERROR = None
except Exception as err:  # pragma: no cover - cloud-only dependency issue
    _WEBRTC_IMPORT_ERROR = err

    class AudioProcessorBase:
        pass

    WebRtcMode = None

    def webrtc_streamer(*args, _err=err, **kwargs):
        raise RuntimeError(f"streamlit-webrtc is unavailable: {_err}")

import storage  # Course/Module/Title persistence (Azure Blob); UI-free helpers.

# Azure Fast Transcription REST API version (GA).
FAST_TRANSCRIPTION_API_VERSION = "2024-11-15"


def get_credential(name: str):
    """Read a credential from wherever it lives, without hard-coding it.

    - On Streamlit Community Cloud, secrets are set in the app dashboard and
      read via st.secrets.
    - Locally, they come from a .env file / environment variable.

    This lets the SAME code run both locally and in the cloud unchanged.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No secrets file (e.g. running locally) — fall through to env vars.
        pass
    return os.getenv(name)


def _cloudflare_ice_servers():
    """Mint short-lived TURN credentials from Cloudflare, if configured.

    Cloudflare's TURN service hands out ephemeral credentials rather than a
    static username/password, so we ask its API for a fresh set on each render.
    Set two secrets to enable it:
      - CLOUDFLARE_TURN_KEY_ID
      - CLOUDFLARE_TURN_API_TOKEN
    Returns Cloudflare's ready-made iceServers list (STUN + TURN), or None when
    not configured or the request fails (so STUN-only still works locally).
    """
    key_id = get_credential("CLOUDFLARE_TURN_KEY_ID")
    api_token = get_credential("CLOUDFLARE_TURN_API_TOKEN")
    if not key_id or not api_token:
        return None
    try:
        resp = requests.post(
            f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}"
            "/credentials/generate-ice-servers",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={"ttl": 86400},  # 24h — comfortably longer than any session
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("iceServers")
    except Exception:
        # Never let a TURN hiccup break page render — STUN is still set.
        return None


def build_rtc_configuration():
    """ICE servers for the real-time (WebRTC) live transcription.

    Always includes a public STUN server. On cloud hosting (Streamlit Cloud) or
    mobile / restrictive networks (e.g. an iPhone on cellular), STUN alone can't
    connect — a TURN relay is required. We support two ways to supply one:
      - Cloudflare TURN, via CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_API_TOKEN
        secrets (preferred; credentials are minted fresh per render), or
      - a static TURN_CONFIG secret (a JSON list of ICE server entries).
    Keeping these in secrets means no credentials in the code, and local dev
    still works with STUN only.
    """
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    cf_ice = _cloudflare_ice_servers()
    if cf_ice:
        ice_servers.extend(cf_ice)

    turn = get_credential("TURN_CONFIG")
    if turn:
        try:
            extra = json.loads(turn) if isinstance(turn, str) else turn
            if isinstance(extra, dict):
                extra = [extra]
            ice_servers.extend(extra)
        except Exception:
            # Bad/missing TURN config shouldn't break the app — STUN still set.
            pass
    return {"iceServers": ice_servers}


def _dependency_messages():
    messages = []
    if _AV_IMPORT_ERROR is not None:
        messages.append(f"PyAV import failed: {_AV_IMPORT_ERROR}")
    if _SPEECH_IMPORT_ERROR is not None:
        messages.append(f"Azure Speech import failed: {_SPEECH_IMPORT_ERROR}")
    if _WEBRTC_IMPORT_ERROR is not None:
        messages.append(f"streamlit-webrtc import failed: {_WEBRTC_IMPORT_ERROR}")
    if globals().get("_IMAGEIO_FFMPEG_ERROR") is not None:
        messages.append(f"imageio-ffmpeg import failed: {_IMAGEIO_FFMPEG_ERROR}")
    return messages


# load_dotenv() reads a local ".env" file into the environment for local runs.
# The secret key is NEVER written in this code — it only lives in .env locally
# or in the Streamlit Cloud secrets dashboard when deployed.
load_dotenv()
AZURE_SPEECH_KEY = get_credential("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = get_credential("AZURE_SPEECH_REGION")


def convert_to_wav(input_path: str) -> str:
    """Convert an audio file (e.g. MP3) into a 16 kHz mono PCM WAV file.

    Azure's file input only understands PCM WAV, so for other formats like MP3
    we first convert. We use a bundled ffmpeg that comes with the
    imageio-ffmpeg pip package, so you don't have to install ffmpeg yourself.
    Returns the path to the new WAV file.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    wav_path = input_path + ".converted.wav"
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Show the tail of ffmpeg's error so problems are easy to diagnose.
        raise RuntimeError(f"Could not convert audio to WAV: {result.stderr[-300:]}")
    return wav_path


def transcribe_audio_file(file_path: str, key: str, region: str,
                          locale: str = "en-US") -> str:
    """Transcribe a whole audio file using Azure's Fast Transcription API.

    WHY FAST TRANSCRIPTION (instead of real-time recognition):
    Real-time recognition processes audio roughly at playback speed, which is
    painfully slow for a 90-minute lecture. The Fast Transcription REST API is
    built for pre-recorded files: it processes them much faster than real-time
    and returns the complete transcript in a single response.

    We send the audio with one HTTP POST and read the joined transcript back
    out of the JSON response. Raises RuntimeError if Azure returns an error.
    """
    url = (
        f"https://{region}.api.cognitive.microsoft.com"
        f"/speechtotext/transcriptions:transcribe"
        f"?api-version={FAST_TRANSCRIPTION_API_VERSION}"
    )
    # "definition" tells Azure which language(s) to expect.
    definition = json.dumps({"locales": [locale]})

    with open(file_path, "rb") as audio:
        files = {
            "audio": (os.path.basename(file_path), audio, "audio/wav"),
            "definition": (None, definition, "application/json"),
        }
        response = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": key},
            files=files,
            timeout=600,  # allow plenty of time for long lectures
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Azure returned HTTP {response.status_code}: {response.text[:300]}"
        )

    # The response groups the recognized text into "combinedPhrases".
    data = response.json()
    phrases = data.get("combinedPhrases", [])
    return " ".join(p.get("text", "") for p in phrases).strip()


class AzureLiveTranscriber(AudioProcessorBase):
    """Streams microphone audio (captured in the browser) to Azure for live STT.

    HOW LIVE MODE WORKS ON ANY DEVICE:
    streamlit-webrtc captures audio in the user's BROWSER and sends it to the
    server as a stream of audio "frames". For each frame we:
      1. resample it to 16 kHz mono 16-bit PCM (the format Azure expects), and
      2. push it into an Azure PushAudioInputStream.
    A SpeechRecognizer reads from that stream and fires events as it recognizes
    speech, which we collect into the transcript. Because the mic lives in the
    browser, this works on phones, tablets, and laptops — including in the cloud.
    """

    def __init__(self):
        if speechsdk is None or av is None:
            raise RuntimeError(
                "Live transcription dependencies are unavailable: "
                + "; ".join(_dependency_messages())
            )
        self._lock = threading.Lock()
        self._final = []     # finalized sentences
        self._partial = ""   # the phrase currently being recognized
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=16000
        )

        # Diagnostics so we can see whether audio is actually flowing.
        self.frames_in = 0     # audio frames received from the browser
        self.bytes_pushed = 0  # PCM bytes sent to Azure
        self.errors = []       # any errors from Azure or frame handling

        # Lifecycle: continuous recognition can stop on its own after a silence
        # or a transient error. We auto-restart it (see _on_session_stopped) so
        # a pause doesn't permanently kill live transcription. `_stopping` tells
        # the restart logic to stand down once we're intentionally shutting down.
        self._stopping = False
        self._bytes_at_last_start = 0

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=audio_format
        )
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION
        )
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_session_stopped)
        self._recognizer.start_continuous_recognition()

    def _on_recognizing(self, evt):
        # Interim (not-yet-final) words, updated as you speak.
        with self._lock:
            self._partial = evt.result.text

    def _on_recognized(self, evt):
        # A finalized chunk of speech.
        if (evt.result.reason == speechsdk.ResultReason.RecognizedSpeech
                and evt.result.text):
            with self._lock:
                self._final.append(evt.result.text)
                self._partial = ""

    def _on_canceled(self, evt):
        # Surface Azure-side problems (bad key, etc.) instead of failing silently.
        if evt.reason == speechsdk.CancellationReason.Error:
            with self._lock:
                self.errors.append(str(evt.error_details))

    def _on_session_stopped(self, evt):
        # Azure ended this recognition session — typically after a long silence,
        # an end-of-stream, or a transient error. Restart it so audio that
        # resumes after a pause is still transcribed, instead of going dead.
        #
        # Guard against tight restart loops: only restart if audio actually
        # flowed since the last start. A session that stops having pushed no new
        # bytes (e.g. a bad key that fails immediately) would otherwise respawn
        # forever; this lets it stop instead.
        with self._lock:
            if self._stopping:
                return
            made_progress = self.bytes_pushed > self._bytes_at_last_start
            self._bytes_at_last_start = self.bytes_pushed
            self._partial = ""
        if not made_progress:
            return
        try:
            self._recognizer.start_continuous_recognition_async()
        except Exception as err:
            with self._lock:
                self.errors.append(f"recognition restart: {err}")

    async def recv_queued(self, frames):
        # Called by streamlit-webrtc with a batch of incoming audio frames.
        # MUST be async: the library schedules it with loop.create_task(), which
        # requires a coroutine. The work inside is fast (resample + push), so
        # running it inline here is fine.
        try:
            for frame in frames:
                frame.pts = None  # let the resampler manage timing
                for resampled in self._resampler.resample(frame):
                    data = resampled.to_ndarray().tobytes()
                    self._push_stream.write(data)
                    self.bytes_pushed += len(data)
            self.frames_in += len(frames)
        except Exception as err:  # don't let a frame error kill the stream silently
            with self._lock:
                self.errors.append(f"frame handling: {err}")
        # MUST return a LIST of frames: the library does out_deque.extend(result).
        # Returning a single frame (or None) crashes the worker after one batch.
        return frames

    def get_status(self):
        with self._lock:
            return self.frames_in, self.bytes_pushed, list(self.errors)

    def get_transcript(self) -> str:
        with self._lock:
            text = " ".join(self._final)
            if self._partial:
                text = (text + " " + self._partial).strip()
            return text

    def get_segments(self):
        """Return (finalized_segments_copy, current_partial).

        The caller accumulates finalized segments into Streamlit session state
        so the transcript survives this processor being torn down and rebuilt
        on a reconnect (which is what used to wipe everything on a pause).
        """
        with self._lock:
            return list(self._final), self._partial

    def stop(self):
        with self._lock:
            self._stopping = True  # tell _on_session_stopped not to restart
        try:
            self._recognizer.stop_continuous_recognition()
            self._push_stream.close()
        except Exception:
            pass

    def __del__(self):
        self.stop()


def render_live_tab():
    """Live audio: a reliable in-browser recorder, plus experimental real-time."""
    st.subheader("Transcribe live audio")

    dependency_messages = _dependency_messages()
    if dependency_messages:
        st.error(
            "Live transcription dependencies failed to load on this server: "
            + " | ".join(dependency_messages)
        )
        st.info("The Upload tab can still work if Azure Speech is configured.")
        return

    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        st.error(
            "Azure credentials are not set. Configure your .env file (local) or "
            "Streamlit secrets (cloud) first."
        )
        return

    rec_tab, rt_tab = st.tabs(["🎙️ Record (recommended)", "⚡ Real-time (beta)"])
    with rec_tab:
        render_record_section()
    with rt_tab:
        render_realtime_section()


def render_record_section():
    """Record from the browser mic, then transcribe. Works on any device."""
    st.write(
        "Tap the microphone to start recording, tap again to stop, then click "
        "**Transcribe**. Works on phones, tablets, and laptops — including the cloud."
    )
    recording = st.audio_input("Record your lecture", key="live_recorder")
    if recording is None:
        return
    # Ask where to save BEFORE transcribing (no-op if storage isn't configured).
    dest = pick_destination("rec")
    if not st.button("Transcribe recording", type="primary", key="live_rec_btn"):
        return
    try:
        with st.spinner("Transcribing…"):
            transcript = transcribe_uploaded_audio(recording)
    except RuntimeError as err:
        st.error(f"Transcription failed: {err}")
        return
    if transcript:
        st.success("Done!")
        st.text_area("Transcript", transcript, height=300, key="live_rec_text")
        save_to_library(dest, transcript, recording, "recording.wav")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=transcript,
            file_name="recording_transcript.txt",
            mime="text/plain",
            key="live_rec_dl",
        )
    else:
        st.warning(
            "No speech was recognized. Try again, speaking a bit louder or "
            "closer to the microphone."
        )


def render_realtime_section():
    """Experimental real-time word-by-word transcription via WebRTC."""
    st.write("Click **START**, allow microphone access, and speak. Words appear as you talk.")
    st.caption(
        "Experimental: streams audio from your browser in real time. If it won't "
        "connect (e.g. on a phone or restrictive network), use the Record tab instead."
    )

    # Build the ICE config once so we can both use it and report on it. On
    # Streamlit Cloud a TURN relay is mandatory; if none made it into the config
    # (e.g. the Cloudflare secrets aren't wired up), say so up front instead of
    # letting it silently fail to connect.
    rtc_config = build_rtc_configuration()
    turn_urls = [
        u
        for server in rtc_config["iceServers"]
        for u in ([server["urls"]] if isinstance(server.get("urls"), str)
                  else server.get("urls", []))
        if str(u).startswith(("turn:", "turns:"))
    ]
    if turn_urls:
        st.caption(f"TURN relay configured ✓ ({len(turn_urls)} relay URLs).")
    else:
        st.warning(
            "No TURN relay is configured — on Streamlit Cloud the connection "
            "can't form without one. Set the CLOUDFLARE_TURN_KEY_ID and "
            "CLOUDFLARE_TURN_API_TOKEN secrets (exact names)."
        )

    # The webrtc component shows its own START / STOP buttons and handles the
    # browser microphone permission prompt.
    ctx = webrtc_streamer(
        key="live-transcribe",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AzureLiveTranscriber,
        media_stream_constraints={"audio": True, "video": False},
        # STUN + TURN (from secrets). rtc_configuration is a shorthand that
        # configures BOTH the browser and the server-side peer.
        rtc_configuration=rtc_config,
        async_processing=True,
    )

    transcript_box = st.empty()

    if ctx.state.playing:
        st.info("🔴 Listening… speak now.")
        if ctx.audio_processor:
            # Accumulate finalized speech into session state so it OUTLIVES the
            # audio processor. On a pause, a flaky network can drop the WebRTC
            # connection; streamlit-webrtc then builds a fresh processor with an
            # empty transcript. We therefore never read the whole transcript off
            # the (possibly brand-new) processor — we append only its NOT-yet-
            # consumed finalized segments to what we've already banked.
            finals, partial = ctx.audio_processor.get_segments()

            # A new processor object means a (re)connection happened: reset how
            # many of its segments we've consumed, but KEEP the banked text.
            proc_id = id(ctx.audio_processor)
            if st.session_state.get("rt_proc_id") != proc_id:
                st.session_state["rt_proc_id"] = proc_id
                st.session_state["rt_consumed"] = 0

            consumed = st.session_state.get("rt_consumed", 0)
            new_segments = finals[consumed:]
            if new_segments:
                banked = st.session_state.get("live_transcript", "")
                banked = (banked + " " + " ".join(new_segments)).strip()
                st.session_state["live_transcript"] = banked
                st.session_state["rt_consumed"] = len(finals)

            # Show banked text plus the live partial (partial is never banked —
            # it gets replaced by a finalized segment once Azure commits it).
            banked = st.session_state.get("live_transcript", "")
            text = (banked + " " + partial).strip() if partial else banked
            # Use markdown (not a keyed text_area): a widget with a key ignores
            # its `value` after first render, so a live-updating text_area would
            # stay frozen on its initial (empty) value.
            transcript_box.markdown(
                f"**Live transcript:**\n\n{text if text else '_(listening…)_'}"
            )

            # Diagnostics: shows exactly where the pipeline breaks when no words
            # appear. frames_in=0 means the browser audio isn't reaching the
            # server (usually a TURN/media relay problem on restrictive
            # networks); bytes_pushed>0 with no transcript points at Azure.
            frames_in, bytes_pushed, errors = ctx.audio_processor.get_status()
            st.caption(
                f"Audio in: {frames_in} frames · {bytes_pushed:,} bytes sent to Azure"
            )
            if frames_in == 0:
                st.info(
                    "Waiting for audio… the first few seconds after START are "
                    "normal while the connection negotiates. If frames stay at 0 "
                    "after ~10s of speaking, the media relay isn't working — check "
                    "that the TURN relay shows configured above."
                )
            for err in errors:
                st.error(f"Azure error: {err}")
        else:
            st.warning("Connecting… if this persists, the audio stream isn't reaching the server.")
        # Refresh once a second so newly recognized words appear.
        time.sleep(1)
        st.rerun()
    else:
        final = st.session_state.get("live_transcript", "")
        transcript_box.markdown(
            f"**Live transcript:**\n\n{final if final else '_(nothing recorded yet)_'}"
        )
        if final:
            st.download_button(
                "⬇️ Download transcript (.txt)",
                data=final,
                file_name="live_transcript.txt",
                mime="text/plain",
                key="rt_dl",
            )
            # The transcript is kept across reconnects, so it also persists after
            # you STOP. Clear it before starting a genuinely new recording so the
            # next session doesn't append onto this one.
            if st.button("🗑️ Clear transcript", key="rt_clear"):
                for k in ("live_transcript", "rt_consumed", "rt_proc_id"):
                    st.session_state.pop(k, None)
                st.rerun()


def transcribe_uploaded_audio(uploaded_file) -> str:
    """Save an uploaded OR recorded audio file, normalize it, and transcribe it.

    Shared by the Upload tab and the Live "Record" tab. Writes the audio to a
    temp file, converts it to 16 kHz mono WAV with ffmpeg, sends it to Azure
    Fast Transcription, and always cleans up the temp files.
    """
    temp_path = None
    wav_path = None
    try:
        name = getattr(uploaded_file, "name", "") or "audio.wav"
        suffix = os.path.splitext(name)[1].lower() or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        wav_path = convert_to_wav(temp_path)
        return transcribe_audio_file(
            wav_path, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION
        )
    finally:
        # Best-effort cleanup; never crash if a handle is briefly still held.
        for path in {temp_path, wav_path}:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# Sentinel shown in each layer's dropdown to start creating a new folder.
_NEW_FOLDER = "➕ Create new…"


def _layer_picker(label: str, existing, key: str):
    """One layer of the destination picker: choose an existing folder OR type a
    new name. Returns the chosen/typed name, or None if nothing is set yet."""
    choice = st.selectbox(label, existing + [_NEW_FOLDER], key=f"{key}_sel")
    if choice == _NEW_FOLDER:
        typed = st.text_input(f"New {label.lower()} name", key=f"{key}_new")
        return typed.strip() or None
    return choice


def pick_destination(key_prefix: str):
    """Ask where to save: Course -> Module -> Title, each with inline create.

    Shown BEFORE transcribing so the user chooses a destination up front.
    Returns a (course, module, title) tuple once all three are set, otherwise
    None (including when storage isn't configured, so callers can skip saving).
    """
    if not storage.storage_available():
        st.caption(
            "💾 Saving is off — add an AZURE_STORAGE_CONNECTION_STRING secret to "
            "store this in your Course / Module / Title library."
        )
        return None

    st.markdown("**Where should this be saved?**")
    course = _layer_picker("Course", storage.list_courses(), f"{key_prefix}_c")
    if not course:
        return None
    module = _layer_picker("Module", storage.list_modules(course), f"{key_prefix}_m")
    if not module:
        return None
    title = _layer_picker("Title", storage.list_titles(course, module), f"{key_prefix}_t")
    if not title:
        return None
    return course, module, title


def _audio_bytes_and_name(audio_file, fallback_name: str):
    """Pull raw bytes + a filename out of an uploaded/recorded audio object."""
    if hasattr(audio_file, "getvalue"):
        data = audio_file.getvalue()
    else:
        audio_file.seek(0)
        data = audio_file.read()
    name = getattr(audio_file, "name", "") or fallback_name
    return data, name


def save_to_library(dest, transcript: str, audio_file, audio_fallback_name: str):
    """Persist the transcript and its audio into the chosen Course/Module/Title.

    `dest` is the (course, module, title) tuple from pick_destination, or None
    (saving disabled / not chosen) in which case this is a no-op.
    """
    if not dest:
        return
    course, module, title = dest
    try:
        storage.save_transcript(course, module, title, transcript)
        if audio_file is not None:
            data, name = _audio_bytes_and_name(audio_file, audio_fallback_name)
            storage.save_bytes(
                course, module, title, name, data,
                getattr(audio_file, "type", None),
            )
        st.success(f"💾 Saved to {course} / {module} / {title}")
    except Exception as err:
        st.error(f"Couldn't save to library: {err}")


def render_library_tab():
    """Browse the Course/Module/Title tree and download saved files."""
    st.subheader("📚 Library")
    if not storage.storage_available():
        st.info(
            "The library isn't enabled. Add an AZURE_STORAGE_CONNECTION_STRING "
            "secret (Azure Storage account → Access keys → Connection string) to "
            "save and browse transcripts and audio here."
        )
        return

    # Create folders ahead of time, at any layer.
    with st.expander("➕ Create a folder"):
        new_course = st.text_input("Course", key="lib_new_course")
        new_module = st.text_input("Module (optional)", key="lib_new_module")
        new_title = st.text_input("Title (optional)", key="lib_new_title")
        if st.button("Create folder", key="lib_create_btn"):
            if not new_course.strip():
                st.warning("A course name is required.")
            else:
                try:
                    storage.create_folder(
                        new_course, new_module or None, new_title or None
                    )
                    st.success("Folder created.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Couldn't create folder: {err}")

    courses = storage.list_courses()
    if not courses:
        st.write("Nothing saved yet. Record or upload a lecture, choose a "
                 "destination, and it'll appear here.")
        return

    course = st.selectbox("Course", courses, key="lib_course")
    modules = storage.list_modules(course)
    if not modules:
        st.write("_(no modules in this course yet)_")
        return
    module = st.selectbox("Module", modules, key="lib_module")
    titles = storage.list_titles(course, module)
    if not titles:
        st.write("_(no titles in this module yet)_")
        return
    title = st.selectbox("Title", titles, key="lib_title")

    files = storage.list_files(course, module, title)
    if not files:
        st.write("_(this title has no files yet)_")
        return
    st.write(f"**Files in {course} / {module} / {title}:**")
    for fname, size, blob_name in files:
        data = storage.read_blob(blob_name)
        if fname == "transcript.txt":
            with st.expander("📄 transcript.txt"):
                st.text_area(
                    "Transcript", data.decode("utf-8", "replace"),
                    height=200, key=f"lib_txt_{blob_name}",
                )
        st.download_button(
            f"⬇️ {fname} ({size:,} bytes)",
            data=data, file_name=fname, key=f"lib_dl_{blob_name}",
        )


def render_upload_tab():
    """UI for uploading a pre-recorded lecture and transcribing it."""
    st.subheader("Upload a pre-recorded lecture")
    st.write("Upload a **WAV** or **MP3** file and click *Transcribe*.")

    # Uses Azure Fast Transcription, which handles long lectures quickly.
    # Azure's limit is about 2 hours per file, so split anything longer.
    st.info(
        "ℹ️ Handles long lectures (up to ~2 hours per file). MP3 files and "
        "various WAV formats are converted automatically before transcribing."
    )

    # WHAT THE UPLOADED FILE DOES:
    # The file uploader lets you pick an audio file from your computer. The
    # file is held in memory by Streamlit until we save it for Azure to read.
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is None:
        return

    # Let the user listen back to confirm they picked the right file.
    st.audio(uploaded_file)

    # Ask where to save BEFORE transcribing (no-op if storage isn't configured).
    dest = pick_destination("upload")

    if not st.button("Transcribe", type="primary"):
        return

    if speechsdk is None:
        st.error(
            "Azure Speech could not be imported on this server: "
            f"{_SPEECH_IMPORT_ERROR}"
        )
        return

    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        st.error("Azure credentials are not set. Configure your .env file first.")
        return

    try:
        with st.spinner("Transcribing… long lectures process quickly here."):
            transcript = transcribe_uploaded_audio(uploaded_file)
    except RuntimeError as err:
        st.error(f"Transcription failed: {err}")
        return

    if transcript:
        st.success("Done!")
        st.text_area("Transcript", transcript, height=300, key="upload_text")
        save_to_library(dest, transcript, uploaded_file, "audio.wav")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain",
            key="upload_dl",
        )
    else:
        st.warning(
            "No speech was recognized. Make sure the file is a clear recording."
        )


def main():
    st.set_page_config(page_title="Lecture Audio Transcriber", page_icon="🎙️")
    st.title("🎙️ Lecture Audio Transcriber")
    st.caption("Transcribe lecture audio with Azure Speech-to-Text")

    # Show a clear message about whether credentials are configured
    # (without ever printing the secret key itself).
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        st.warning(
            "Azure credentials are not set. Copy `.env.example` to `.env` and fill "
            "in `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`, then restart the app."
        )
    else:
        st.success(f"Azure Speech configured (region: {AZURE_SPEECH_REGION})")

    tab_upload, tab_live, tab_library = st.tabs(
        ["📁 Upload audio", "🎤 Live audio", "📚 Library"]
    )

    with tab_upload:
        render_upload_tab()

    with tab_live:
        render_live_tab()

    with tab_library:
        render_library_tab()


if __name__ == "__main__":
    main()
