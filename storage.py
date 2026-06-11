"""
Persistent storage for transcripts and audio, organized as a three-layer
hierarchy: Course -> Module -> Title.

WHY AZURE BLOB STORAGE (and not local folders):
Streamlit Community Cloud runs on an EPHEMERAL filesystem — anything the app
writes to local disk is erased on every reboot/redeploy. Azure Blob Storage
persists across restarts, so saved work survives. Blob storage is technically
flat (no real folders), but we model the hierarchy with "/" in blob names:

    <course>/<module>/<title>/transcript.txt
    <course>/<module>/<title>/<original-audio-filename>

A folder you create before saving anything into it is represented by a tiny
empty marker blob named ".keep", so it still shows up in the listings.

This module is intentionally UI-free: it knows nothing about Streamlit widgets,
so the storage logic stays separate from the app's screens.
"""

import mimetypes
import os

import streamlit as st
from azure.storage.blob import ContainerClient, ContentSettings

# A single container holds the whole Course/Module/Title tree.
CONTAINER_NAME = "transcripts"
# Empty marker blob that makes a freshly created (still-empty) folder visible.
FOLDER_MARKER = ".keep"


def _get_credential(name: str):
    """Read a secret from Streamlit secrets (cloud) or env vars (local).

    Mirrors app.py's get_credential so this module can stand alone without
    importing the app (which would create a circular import).
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def storage_available() -> bool:
    """True if a storage connection string is configured."""
    return bool(_get_credential("AZURE_STORAGE_CONNECTION_STRING"))


@st.cache_resource(show_spinner=False)
def _get_container(conn_str: str) -> ContainerClient:
    """Return a ContainerClient, creating the container on first use.

    Cached per connection string so we don't recreate the client (or re-check
    the container) on every Streamlit rerun.
    """
    client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)
    try:
        client.create_container()
    except Exception:
        # Container already exists — that's the normal case after the first run.
        pass
    return client


def _container():
    """The active ContainerClient, or None when storage isn't configured."""
    conn = _get_credential("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return None
    return _get_container(conn)


def _safe(segment: str) -> str:
    """Sanitize one path segment so it can't break out of its folder level."""
    return segment.strip().replace("/", "-").replace("\\", "-")


def _child_folders(prefix: str):
    """List the immediate sub-folder names directly under `prefix`.

    `prefix` is either "" (top level) or ends with "/". We ask Azure to "walk"
    blobs with a "/" delimiter, which groups everything below a folder into a
    single prefix entry (whose name ends with "/"). Plain blobs sitting at this
    level (e.g. a .keep marker) are ignored — only sub-folders are returned.
    """
    container = _container()
    if container is None:
        return []
    names = []
    for item in container.walk_blobs(name_starts_with=prefix, delimiter="/"):
        name = item.name
        if name.endswith("/"):
            child = name[len(prefix):].rstrip("/")
            if child:
                names.append(child)
    return sorted(names)


def list_courses():
    """All course names."""
    return _child_folders("")


def list_modules(course: str):
    """All module names within a course."""
    return _child_folders(f"{_safe(course)}/")


def list_titles(course: str, module: str):
    """All title names within a course/module."""
    return _child_folders(f"{_safe(course)}/{_safe(module)}/")


def create_folder(course: str, module: str = None, title: str = None):
    """Create an (empty) folder at any layer by writing a .keep marker blob."""
    container = _container()
    if container is None:
        raise RuntimeError("Storage is not configured.")
    parts = [_safe(course)]
    if module:
        parts.append(_safe(module))
    if title:
        parts.append(_safe(title))
    marker = "/".join(parts) + "/" + FOLDER_MARKER
    container.upload_blob(marker, b"", overwrite=True)


def _title_prefix(course: str, module: str, title: str) -> str:
    return f"{_safe(course)}/{_safe(module)}/{_safe(title)}/"


def save_bytes(course: str, module: str, title: str, filename: str,
               data: bytes, content_type: str = None) -> str:
    """Store a file inside a Course/Module/Title folder. Returns the blob name."""
    container = _container()
    if container is None:
        raise RuntimeError("Storage is not configured.")
    if content_type is None:
        content_type = mimetypes.guess_type(filename)[0]
    settings = ContentSettings(content_type=content_type) if content_type else None
    blob_name = _title_prefix(course, module, title) + _safe(filename)
    container.upload_blob(
        blob_name, data, overwrite=True, content_settings=settings
    )
    return blob_name


def save_transcript(course: str, module: str, title: str, text: str) -> str:
    """Store the transcript text as transcript.txt."""
    return save_bytes(
        course, module, title, "transcript.txt",
        text.encode("utf-8"), "text/plain",
    )


def list_files(course: str, module: str, title: str):
    """Real files in a title folder as (filename, size_bytes, blob_name).

    Skips the .keep marker and anything in a deeper sub-folder.
    """
    container = _container()
    if container is None:
        return []
    prefix = _title_prefix(course, module, title)
    out = []
    for blob in container.list_blobs(name_starts_with=prefix):
        fname = blob.name[len(prefix):]
        if not fname or fname == FOLDER_MARKER or "/" in fname:
            continue
        out.append((fname, blob.size, blob.name))
    return sorted(out)


def read_blob(blob_name: str) -> bytes:
    """Download a blob's full contents by its name."""
    container = _container()
    if container is None:
        return b""
    return container.download_blob(blob_name).readall()
