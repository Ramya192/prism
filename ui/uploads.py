# ui/uploads.py — the one place every file uploader in the UI goes through,
# so the public-demo notice (and, later, an upload on/off switch) lives in a
# single spot instead of being copied beside each st.file_uploader call.

import os

import streamlit as st

DEMO_NOTICE = (
    "Demo only: please try the sample files, not real personal or financial documents. "
    "Uploads are stored under your browser session and may be deleted."
)


def samples_only() -> bool:
    """PRISM_SAMPLES_ONLY=1 turns every uploader off, leaving the sample-file
    pickers; off by default."""
    return os.getenv("PRISM_SAMPLES_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def demo_file_uploader(label: str, **kwargs):
    """st.file_uploader plus the demo notice underneath. Takes and returns
    exactly what st.file_uploader does; with PRISM_SAMPLES_ONLY set it shows a
    short message instead and returns what an empty uploader would."""
    if samples_only():
        st.info("Uploads are turned off on this demo. Please pick one of the sample files instead.")
        return [] if kwargs.get("accept_multiple_files") else None
    widget = st.file_uploader(label, **kwargs)
    st.caption(f"🔒 {DEMO_NOTICE}")
    return widget
