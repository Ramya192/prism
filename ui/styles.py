# ui/styles.py — global CSS (applies to both the landing page and every
# workspace) plus the top navbar. Split out of streamlit_app.py, which had
# grown to 1483 lines; the landing page's OWN CSS (used nowhere else)
# stays local to ui/landing.py instead of living here.

import streamlit as st

GLOBAL_CSS = """
<style>
.block-container { padding-top: 3.6rem !important; }

.st-key-top_navbar {
    display: flex !important; flex-direction: row !important; justify-content: space-between !important;
    align-items: center !important; padding: 4px 0 16px; margin-bottom: 8px;
    border-bottom: 1px solid #e3e6ec;
}
.st-key-top_navbar > .stElementContainer { width: auto !important; flex: 0 0 auto !important; }
.st-key-top_navbar .navbar-logo { display: flex; align-items: center; gap: 12px; }
.st-key-top_navbar .navbar-mark {
    width: 40px; height: 40px; border-radius: 10px; background: #2f5fdb; color: #fff;
    display: flex; align-items: center; justify-content: center;
    line-height: 1; flex-shrink: 0;
}
.st-key-top_navbar .navbar-wordmark {
    font-size: 27px; font-weight: 800; letter-spacing: 0.04em; color: #2f5fdb; text-transform: uppercase;
}
.st-key-top_navbar .navbar-author { font-size: 16px; font-weight: 500; color: #626977; }

/* Dismissible workspace notice banners (PayNexus-style) */
.st-key-banner_info, .st-key-banner_warn {
    border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;
}
.st-key-banner_info { background: #eef4fc; border: 1px solid #cfe0f7; }
.st-key-banner_warn { background: #fff8e6; border: 1px solid #f3dfa0; }
.banner-title { font-weight: 700; font-size: 13.5px; color: #191d24; }
.banner-body { font-size: 12.5px; color: #626977; margin-top: 3px; }
.st-key-banner_info .stButton>button, .st-key-banner_warn .stButton>button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: #8993a6 !important; padding: 0 !important; font-size: 14px !important;
}

/* Floating chat bubble + drawer (replaces the old docked side panel) */
.st-key-chat_fab {
    position: fixed !important; bottom: 28px; right: 28px; z-index: 9999;
    width: 56px !important; flex: 0 0 auto !important;
}
.st-key-chat_fab .stButton>button {
    width: 56px; height: 56px; border-radius: 50%; background: #2f5fdb !important; color: #fff !important;
    font-size: 22px; box-shadow: 0 4px 14px rgba(20,24,32,.25); border: none !important;
}
.st-key-chat_drawer {
    position: fixed !important; top: 132px; bottom: 28px; right: 28px; z-index: 9998;
    width: 440px !important; max-width: calc(100vw - 40px); overflow-y: auto;
    background: #ffffff; border: 1px solid #e3e6ec; border-radius: 16px;
    box-shadow: 0 8px 30px rgba(20,24,32,.18); padding: 16px;
}
.chat-drawer-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; padding-top: 4px; }
.st-key-chat_max_btn_wrap .stButton>button {
    border: none !important; background: transparent !important; box-shadow: none !important;
    color: #8993a6 !important; padding: 0 !important; font-size: 16px !important;
}
.st-key-chat_max_btn_wrap .stButton>button:hover { color: #2f5fdb !important; }

/* "What happens next" strip */
.next-steps {
    display: flex; gap: 18px; flex-wrap: wrap; justify-content: center;
    font-size: 11.5px; color: #8993a6; margin-top: 16px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
}
.next-steps span { white-space: nowrap; }

/* Architecture flow diagram */
.arch-flow { text-align: center; padding: 18px 10px; }
.arch-flow .flow-node {
    display: inline-block; border: 1px solid #e3e6ec; border-radius: 10px;
    background: #f6f7f9; padding: 6px 16px; font-size: 12.5px; font-weight: 600; color: #191d24;
}
.arch-flow .flow-root { background: #2f5fdb; color: #fff; border-color: #2f5fdb; }
.arch-flow .flow-result { background: #191d24; color: #fff; border-color: #191d24; }
.arch-flow .flow-arrow { color: #b7bec9; font-size: 13px; margin: 4px 0; }
.arch-flow .flow-detected { font-size: 12px; color: #626977; margin: 2px 0; }
.arch-flow .flow-branches { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }
.arch-flow .flow-branch {
    border: 1px solid #cfe0f7; background: #eef4fc; border-radius: 10px;
    padding: 8px 14px; font-size: 12px; font-weight: 600; color: #191d24;
}
.arch-flow .flow-sub { font-size: 10.5px; font-weight: 500; color: #626977; }

/* Config-driven panel — real values pulled from configs/<id>.yaml */
.config-panel {
    border: 1px solid #e3e6ec; border-radius: 12px; background: #f6f7f9;
    padding: 14px 18px; height: 100%;
}
.config-panel-title {
    font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; font-weight: 700;
    letter-spacing: .06em; color: #8993a6; margin-bottom: 10px; text-transform: uppercase;
}
.config-row { display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; font-size: 12.5px; border-bottom: 1px dashed #e3e6ec; }
.config-row:last-child { border-bottom: none; }
.config-key { color: #626977; }
.config-val { color: #191d24; font-weight: 600; text-align: right; }

/* Result "wow" card */
.st-key-analysis_card {
    border: 1px solid #e3e6ec; border-radius: 14px; padding: 16px 18px; margin-top: 8px;
    background: linear-gradient(180deg, #fbfcfe 0%, #ffffff 100%);
}
.analysis-title {
    font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; font-weight: 700;
    letter-spacing: .06em; color: #8993a6; margin-bottom: 8px; text-transform: uppercase;
}
.pipeline-checklist {
    display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: #2f8a3e;
    margin: 8px 0 4px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
}

/* Explainability breakdown */
.explain-panel { margin-top: 10px; border-top: 1px solid #e3e6ec; padding-top: 10px; }
.explain-title { font-size: 11px; font-weight: 700; color: #8993a6; letter-spacing: .04em; margin-bottom: 6px; }
.explain-row { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.explain-tier {
    font-size: 11.5px; font-weight: 700; color: #2f5fdb; background: #eef4fc;
    border-radius: 6px; padding: 2px 8px; white-space: nowrap;
}
.explain-reason { font-size: 12.5px; color: #191d24; }

.stButton>button { transition: all .15s ease; }
</style>
"""

PRISM_ICON_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<line x1="0" y1="13" x2="8.5" y2="13" stroke="#ffffff" stroke-width="1.7" stroke-linecap="round"/>
<polygon points="12,3 21,19 3,19" fill="#ffffff" fill-opacity="0.95"/>
<line x1="14.3" y1="13.6" x2="23.5" y2="7.5" stroke="#ff5c5c" stroke-width="1.4" stroke-linecap="round"/>
<line x1="14.9" y1="15.1" x2="24" y2="12.2" stroke="#ffb703" stroke-width="1.4" stroke-linecap="round"/>
<line x1="15.1" y1="16.7" x2="24" y2="17" stroke="#2ec4b6" stroke-width="1.4" stroke-linecap="round"/>
<line x1="14.6" y1="18.1" x2="22.5" y2="21.5" stroke="#3a86ff" stroke-width="1.4" stroke-linecap="round"/>
</svg>"""


def render_top_navbar() -> None:
    """Brand mark top-left, author top-right. Rendered once, above both
    step 1 (landing) and step 2 (workspace), so the branding doesn't
    reset/duplicate when a domain is confirmed."""
    with st.container(key="top_navbar"):
        st.markdown(
            f"<div class='navbar-logo'><div class='navbar-mark'>{PRISM_ICON_SVG}</div>"
            "<div class='navbar-wordmark'>Prism</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='navbar-author'>Ramya A</div>", unsafe_allow_html=True)
