# streamlit_app.py — Prism's single entry point.
# Flow: a command bar (paste/upload, or jump straight to a domain) ->
# DomainClassifierAgent guesses the domain -> a human confirms or
# overrides -> AgentOrchestrator hands back that domain's Pipeline ->
# that domain's workspace takes over. The confirm step is deliberate,
# not a formality — no pipeline ever runs on the classifier's guess
# alone, and the quick-pick chips below skip the guess, never the
# confirm (they set confirmed_domain directly, same trust level as
# picking from the dropdown after a guess).
#
# Every domain's workspace follows the same shape: tabs for how you
# provide data (upload / manual entry / image input, depending on the
# domain), a verdict card for the auto fraud check, and a floating chat
# bubble (not a tab) that opens a drawer for follow-up questions rather
# than uploading or ingesting anything itself. Every chat answer carries
# citation chips showing whether it drew on the uploaded document, the
# domain's static reference corpus, or both.
#
# This script itself is just page setup + the 3-way step routing
# (landing / batch results / domain workspace); the actual rendering
# lives in ui/ (shared renderers, landing page, batch upload, and one
# module per domain workspace) -- split out once this file crossed 1483
# lines, so each domain's UI has its own file instead of one script
# growing without bound as a 5th domain gets added.

import os

import streamlit as st

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from core.domain_classifier_agent import DomainClassifierAgent

from ui.constants import WORKSPACE_SUBTITLES
from ui.guard import require_access_code
from ui.styles import GLOBAL_CSS, render_top_navbar
from ui.shared import render_config_panel, render_banner
from ui.landing import render_landing
from ui.batch_upload import render_batch_results_view
from ui.domains.banking import render_banking_workspace
from ui.domains.insurance import render_insurance_workspace
from ui.domains.financial_services import render_financial_services_workspace
from ui.domains.payroll_hr import render_payroll_hr_workspace

st.set_page_config(page_title="Prism", page_icon="◈", layout="wide")
require_access_code()   # no-op unless PRISM_ACCESS_CODE is set (ui/guard.py)


@st.cache_resource
def load_core():
    loader = ConfigLoader()
    orchestrator = AgentOrchestrator(loader)
    classifier = DomainClassifierAgent(loader)
    return loader, orchestrator, classifier


config_loader, orchestrator, classifier = load_core()

# ══════════════════════════════════════════════════════════════════════
# Top navbar — brand mark top-left, author top-right. Rendered once,
# above both step 1 (landing) and step 2 (workspace), so the branding
# doesn't reset/duplicate when a domain is confirmed.
# ══════════════════════════════════════════════════════════════════════

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
render_top_navbar()

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Command bar (ui/landing.py) — no sidebar domain list (every
# runnable domain is already one tap away via the quick-pick chips, and a
# permanent "working"/"coming soon" panel added nothing a chip doesn't
# already convey via being enabled/disabled).
# ══════════════════════════════════════════════════════════════════════

if "confirmed_domain" not in st.session_state and "batch_results" not in st.session_state:
    render_landing(config_loader, orchestrator, classifier)

# ══════════════════════════════════════════════════════════════════════
# STEP 2b — batch-upload results, grouped by domain (see
# ui/batch_upload.py's render_batch_upload_section for how a file lands
# here)
# ══════════════════════════════════════════════════════════════════════

elif "batch_results" in st.session_state and "confirmed_domain" not in st.session_state:
    render_batch_results_view(config_loader, orchestrator)

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Domain-specific UI, only after human confirmation
# ══════════════════════════════════════════════════════════════════════

else:
    domain_id = st.session_state["confirmed_domain"]
    domain_config = config_loader.get_domain(domain_id)

    # Configuration panel + LLM-backend banner live in the sidebar, not
    # the main body, so the workspace's own tabs render immediately below
    # the header without scrolling. The sidebar is a structurally
    # separate element from Streamlit's own .block-container, so it's
    # also unaffected by render_floating_chat's padding-right reserve
    # when a chat drawer is open.
    with st.sidebar:
        render_config_panel(domain_config)
        _llm_provider = os.getenv("LLM_PROVIDER", "ollama" if os.getenv("ENV", "local") == "local" else "openai")
        if _llm_provider == "ollama":
            render_banner(
                "llm_backend", "info", "Running on local Ollama",
                f"Model: `{os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}` — document extraction and chat "
                "may be slower here than on the hosted OpenAI backend.",
            )
        elif not os.getenv("OPENAI_API_KEY"):
            render_banner(
                "llm_backend", "warn", "OPENAI_API_KEY is not set",
                f"LLM_PROVIDER is `{_llm_provider}` but no API key is configured — LLM-backed steps "
                "(extraction, chat, classification tiebreak) will fail until one is set.",
            )

    st.subheader(domain_config.name)
    st.caption(f"Running as `{domain_id}` — confirmed by you a moment ago.")
    if domain_id in WORKSPACE_SUBTITLES:
        st.caption(WORKSPACE_SUBTITLES[domain_id])
    change_col, reset_col = st.columns([1, 1])
    with change_col:
        if st.button("← change domain"):
            del st.session_state["confirmed_domain"]
            st.rerun()
    with reset_col:
        if st.button("↺ Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.get("staged_ingest_error"):
        st.error(f"Couldn't auto-ingest the file you staged in step 1: {st.session_state['staged_ingest_error']}")
        del st.session_state["staged_ingest_error"]

    st.divider()

    pipeline = orchestrator.get_pipeline(domain_id)

    if domain_id == "banking":
        render_banking_workspace(pipeline)
    elif domain_id == "insurance":
        render_insurance_workspace(pipeline)
    elif domain_id == "financial_services":
        render_financial_services_workspace(pipeline)
    elif domain_id == "payroll_hr":
        render_payroll_hr_workspace(pipeline)
