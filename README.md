# Lecture Audio Transcriber 🎙️

A simple, beginner-friendly web app for transcribing **lecture audio** using
**Azure Speech-to-Text** and **Streamlit**. Upload a pre-recorded WAV file and
get back a text transcript you can download.

## Features

- 📁 Upload a pre-recorded lecture (WAV or MP3) and get a text transcript
  (MP3 files are converted to WAV automatically)
- 🎤 Live transcription from your microphone, on **any device** — the browser
  captures the mic (via WebRTC) and streams it for transcription, so it works on
  phones, tablets, and laptops, locally or deployed
- ⬇️ Download the transcript as a `.txt` file
- 🔒 No secrets in code — Azure credentials are read from environment variables
  (local) or Streamlit Cloud secrets (deployed)

## Prerequisites

- Python 3.9 or newer
- An [Azure Speech resource](https://portal.azure.com/) (free tier works)
- Your Azure Speech **key** and **region** (this app's region is `westus2`)

## 1. Install Python packages

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Create a `.env` file

Copy the example file and fill in your real Azure key:

```bash
# Windows (PowerShell)
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Then edit `.env`:

```
AZURE_SPEECH_KEY=your_real_key_here
AZURE_SPEECH_REGION=westus2
```

⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore` so
your secret key stays off GitHub.

## 3. Run the app locally

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (usually at http://localhost:8501).

## Deploy to Streamlit Community Cloud (recommended — works on any device)

This is the easiest way to get a public URL anyone can open on any device.

1. **Push this project to GitHub.** From this folder:

   ```bash
   git init
   git add .
   git commit -m "Lecture Audio Transcriber"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   The `.gitignore` keeps your real `.env` and `secrets.toml` out of GitHub.

2. **Create the app.** Go to [share.streamlit.io](https://share.streamlit.io),
   click **New app**, pick your repo/branch, and set the main file to `app.py`.

3. **Add your secrets.** In the app's **Settings → Secrets**, paste:

   ```toml
   AZURE_SPEECH_KEY = "your_real_key_here"
   AZURE_SPEECH_REGION = "westus2"
   ```

   The app reads these via `st.secrets` automatically (see `get_credential` in
   `app.py`) — no code changes needed.

4. **Deploy.** Streamlit installs `requirements.txt` (Python packages) and
   `packages.txt` (system libraries) for you, then launches the app at a public
   `https://<your-app>.streamlit.app` URL.

> **Live mic on phones (iPhone/cellular) needs TURN:** the real-time tab streams
> audio over WebRTC. On mobile/restrictive networks, STUN alone can't connect —
> you must add a **TURN** server. Get free TURN credentials (e.g.
> [metered.ca](https://www.metered.ca/tools/openrelay/), 50 GB/mo free), then add
> a `TURN_CONFIG` secret (a JSON list of ICE servers):
>
> ```toml
> TURN_CONFIG = '[{"urls":["turn:HOST:443?transport=tcp","turn:HOST:80"],"username":"USER","credential":"PASS"}]'
> ```
>
> The app reads it via `build_rtc_configuration()` and merges it with STUN — no
> credentials in code. Without TURN, real-time still works on desktop/same
> network, and the **Record** tab + file upload work everywhere.

> **If the cloud build fails** on a missing system library, read the build log
> and add the named package to `packages.txt`, then push again.

## Deploy to Azure App Service

You can host this app on **Azure App Service** (Linux, Python runtime):

1. **Create the Web App.** In the [Azure Portal](https://portal.azure.com/),
   create an *App Service* with a Python runtime (e.g. Python 3.11+, Linux).
2. **Deploy your code** — for example with the Azure CLI from this folder:

   ```bash
   az webapp up --name <your-app-name> --runtime "PYTHON:3.11"
   ```

   (Or use the VS Code *Azure App Service* extension, GitHub Actions, or ZIP
   deploy — whichever you prefer.)
3. **Set the startup command** (see below) so App Service knows how to launch
   Streamlit.
4. **Add your environment variables** in the portal (see below) so the app can
   reach Azure Speech.

### Azure App Service startup command

In the portal go to **Configuration → General settings → Startup Command** and
enter:

```bash
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

> App Service expects your app to listen on port **8000** and on address
> **0.0.0.0** (all interfaces), which is why these flags are required in the
> cloud but not when running locally.

### Environment variables in Azure App Service Configuration

Do **not** upload your `.env` file to Azure. Instead, in the portal go to
**Configuration → Application settings** and add these as app settings:

| Name                  | Value                  |
| --------------------- | ---------------------- |
| `AZURE_SPEECH_KEY`    | your real Azure key    |
| `AZURE_SPEECH_REGION` | `westus2`              |

App Service injects these as environment variables, and the app reads them the
same way it reads your local `.env` — so no code changes are needed.

## ⚠️ Important limitation

Uploaded files are transcribed with Azure's **Fast Transcription API**, which
handles long lectures quickly (much faster than real-time). Azure's limit is
about **2 hours per file**, so split anything longer into separate uploads.
WAV and MP3 are supported — files are normalized to 16 kHz mono WAV
automatically (using the bundled ffmpeg from the `imageio-ffmpeg` package),
which also keeps large files under Azure's size limit.

## Project structure

```
Transcrybe/
├── app.py                          # The Streamlit web app
├── requirements.txt                # Python dependencies
├── packages.txt                    # System libraries for Streamlit Cloud
├── .env.example                    # Template for local Azure credentials
├── .streamlit/config.toml          # Streamlit settings (e.g. max upload size)
├── .streamlit/secrets.toml.example # Template for Streamlit Cloud secrets
├── .gitignore                      # Files git should ignore (.env, secrets)
└── README.md                       # This file
```

## Upload size limit

Streamlit's default maximum upload is 200 MB. This app raises it to **600 MB**
via `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 600
```

This setting is picked up automatically both locally and on Azure App Service,
so no command-line flags are needed.

## Security

This app **never** hard-codes Azure keys. Credentials are loaded only from:

- a git-ignored `.env` file (local), or
- Streamlit Community Cloud **Secrets** (when deployed there), or
- App Service **Application settings** (Azure deployment).

The `get_credential()` helper in `app.py` checks `st.secrets` first, then falls
back to environment variables, so the same code runs everywhere unchanged.
