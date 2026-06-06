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

import av
import requests
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import imageio_ffmpeg
from dotenv import load_dotenv
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

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
        return frames[-1] if frames else None

    def get_status(self):
        with self._lock:
            return self.frames_in, self.bytes_pushed, list(self.errors)

    def get_transcript(self) -> str:
        with self._lock:
            text = " ".join(self._final)
            if self._partial:
                text = (text + " " + self._partial).strip()
            return text

    def stop(self):
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

    # The webrtc component shows its own START / STOP buttons and handles the
    # browser microphone permission prompt.
    ctx = webrtc_streamer(
        key="live-transcribe",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AzureLiveTranscriber,
        media_stream_constraints={"audio": True, "video": False},
        # Public STUN server helps browsers connect through firewalls/NAT.
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    transcript_box = st.empty()

    if ctx.state.playing:
        st.info("🔴 Listening… speak now.")
        if ctx.audio_processor:
            text = ctx.audio_processor.get_transcript()
            transcript_box.text_area("Live transcript", text, height=300, key="rt_text")
            # Remember it so it stays visible after you click STOP.
            st.session_state["live_transcript"] = text

            # Diagnostics: shows whether audio is reaching the server / Azure.
            frames_in, bytes_pushed, errors = ctx.audio_processor.get_status()
            st.caption(
                f"🔎 audio frames received: {frames_in} · "
                f"PCM bytes sent to Azure: {bytes_pushed:,}"
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
        transcript_box.text_area("Live transcript", final, height=300, key="rt_text_final")
        if final:
            st.download_button(
                "⬇️ Download transcript (.txt)",
                data=final,
                file_name="live_transcript.txt",
                mime="text/plain",
                key="rt_dl",
            )


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

    if not st.button("Transcribe", type="primary"):
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

    tab_upload, tab_live = st.tabs(["📁 Upload audio", "🎤 Live audio"])

    with tab_upload:
        render_upload_tab()

    with tab_live:
        render_live_tab()


if __name__ == "__main__":
    main()
