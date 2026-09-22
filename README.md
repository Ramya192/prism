# ◈ Prism

**Domain-agnostic fraud detection + document intelligence platform.** The
domain is a YAML config parameter, not a hardcoded assumption — Prism
auto-detects what kind of input it's looking at, a human confirms, and the
right multi-agent pipeline runs. Any upload (CSV, PDF, or DOCX), in any
domain, always gets **both** a fraud/anomaly verdict **and** a chat-ready
RAG ingestion — never one or the other based on file format. Add a new
domain by writing a config file and a thin pipeline adapter, not by
forking the app.

Prism supersedes two earlier standalone portfolio projects — **Fraud
Detection** and **FinLens (Document Intelligence)** — by re-platforming
them as domains on a shared config-driven core, instead of separate repos
with duplicated orchestration, UI, and deployment plumbing.

---

## Live Demo

**[https://prism-ramya-aws.duckdns.org](https://prism-ramya-aws.duckdns.org)**

Deployed on AWS EC2 (Amazon Linux 2023, `t3.micro`, free tier) — a single
Dockerized Streamlit service behind [Caddy](https://caddyserver.com) as a
reverse proxy, with an automatically issued and renewed Let's Encrypt TLS
certificate. No ALB, no ECS, no managed load balancer — a deliberately
minimal, cost-free single-instance deployment (see `deploy/README.md` for
the full provisioning runbook).

---

## Architecture

```
User uploads a file (any domain) / describes their data
        │
        ▼
┌───────────────────────────┐
│  DomainClassifierAgent    │  keyword overlap + CSV-header schema
│                            │  sniffing against each domain's known
│                            │  columns; LLM tiebreak only when scores
│                            │  are genuinely close
└───────────────────────────┘
        │  best guess + all scores
        ▼
┌───────────────────────────┐
│  Human confirms/overrides │  ← nothing below this line runs on the
│  (streamlit_app.py)       │    classifier's say-so alone
└───────────────────────────┘
        │  confirmed domain_id
        ▼
┌───────────────────────────┐
│  ConfigLoader              │  configs/<domain>.yaml → validated
│                            │  DomainConfig (Pydantic), ${VAR:default}
│                            │  env interpolation
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  AgentOrchestrator         │  imports & instantiates that domain's
│                            │  Pipeline class, caches it per domain_id
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  domains/<id>/pipeline.py  │  UnifiedDomainPipeline: extraction feeds
│                            │  rules → tiered ML → LLM fraud scoring
│                            │  AND RAG ingestion, on the same upload
└───────────────────────────┘
```

`core/unified_pipeline.py`'s `UnifiedDomainPipeline` is the shared base
every domain builds on: a PDF/DOCX is extracted via LLM, mapped into the
exact record shape the domain's CSV-based detector already expects, and
run through the same rules → ML → LLM cascade a CSV row would hit — while
the document is independently chunked, embedded, and made chattable. A
document with no fraud-scoreable fields (e.g. an HR handbook) degrades
gracefully to chat-only, never a fabricated verdict.

---

## Domains

All 4 business domains are unified — one pipeline per domain, every
upload gets a fraud verdict and RAG chat regardless of file type.

| Domain | Fraud detection | Document chat |
|---|---|---|
| `banking` | Hybrid rules → tiered ML (Tier 1 anonymized PCA schema, Tier 2 real-world-shaped schema) → LLM card/bank-transaction fraud, plus an analyst/router/alert chain and an Isolation Forest drift signal. | Bank statement PDF/DOCX → per-transaction extraction → same fraud engine. RAG grounded in a Regulation E (12 CFR 1005) excerpt. |
| `insurance` | HealthcareDetectorAgent, rules → tiered ML (AUC 0.998 / 0.857, Tier 1 / Tier 2) → LLM claim fraud. | Claim/EOB PDF/DOCX → structured field extraction → same engine. RAG grounded in a CMS Claims Processing Manual (Ch. 26) excerpt. |
| `payroll_hr` | Content-routed between payroll-register fraud (tiered rules → ML → LLM) and job-posting fraud (rules → ML → LLM) — one domain, two fraud shapes, plus HDBSCAN fraud-pattern clustering. | Payslip PDF/DOCX → line-item extraction → same engine; a content-classification gate keeps non-payslip documents (e.g. handbooks) from being force-fit into a fabricated payslip record. RAG grounded in an IRS Publication 15-T excerpt. |
| `financial_services` | FintechDetectorAgent, rules → tiered ML (Tier 1 blockchain-scam schema, Tier 2 exchange/DEX manipulation schema) → LLM. | Wallet/exchange statement PDF/DOCX → per-transaction extraction; every ML tier's required fields are account/backend-level and never derivable from a document, so document-sourced records honestly resolve to LLM-fallback reasoning — still real value over no document capability at all. RAG grounded in a FinCEN CVC guidance (FIN-2019-G001) excerpt. |

Adding a domain means writing `configs/<id>.yaml` (classification hints +
capabilities) and `domains/<id>/pipeline.py` (a `UnifiedDomainPipeline`
subclass) — the core (`ConfigLoader`, `AgentOrchestrator`,
`DomainClassifierAgent`, `streamlit_app.py`'s confirmation flow) doesn't
change.

---

## What the UI does

- **Any-file-type ingestion, per domain** — a workspace's upload tab
  accepts CSV, PDF, or DOCX; every upload yields both a fraud verdict and
  a chattable document. CSV uploads also support multi-file batches
  within one domain, each independently scored and jointly chattable via
  cross-document retrieval.
- **Batch upload across different domains** — a landing-page "Batch
  upload" expander accepts several files at once, classifies each
  independently, lets you override any guess, and — after one combined
  confirm — routes each to its own domain's pipeline. Results are shown
  grouped by domain, with a one-click handoff into that domain's normal
  workspace to keep chatting.
- **Floating chat panel** — a dockable, maximize/minimize chat widget
  available from any workspace, so document Q&A doesn't require leaving
  the fraud-verdict view.
- **Sidebar config panel** — surfaces the active domain's config-driven
  thresholds and capabilities directly from `configs/<domain>.yaml`, so
  the "config, not code" architecture is visible, not just true in the
  source.
- **CSV-header schema sniffing** — a raw CSV column dump (no prose) is
  matched against each domain's known Tier 1/Tier 2 schemas directly, so
  classification doesn't depend on keyword text being present.
- **Drift/outlier signal (banking)** — an Isolation Forest companion to
  the supervised Tier 1 classifier surfaces "does this look like anything
  in the training distribution" as a caveat alongside the fraud verdict.
- **Fraud-pattern clustering (payroll)** — HDBSCAN clustering over the
  payroll anomaly space, validated against 3 injected synthetic fraud
  patterns.
- **Reference-corpus-grounded chat, all 4 domains** — each domain's RAG
  retrieval merges a real, curated regulatory/reference document into the
  same HyDE + multi-query + RRF + rerank ranking as the user's own
  uploaded document, so chat answers can cite domain rules the upload
  itself never mentions.

---

## Evaluation

Every domain ships a real, regenerable eval harness (`domains/<domain>/.../eval_harness.py`)
measuring the *deployed* rules → ML → LLM system against a temporal
holdout at its real base rate — not just raw classifier PR-AUC. All
support `--with-llm-sample` for a small, honestly-labeled stratified
sample of real LLM-tier calls, reported separately from the free
deterministic-layer numbers.

| Domain / tier | Holdout size | Base rate | Deterministic-layer F1 |
|---|---|---|---|
| Banking Tier 1 | 85,443 | 0.13% | 0.234 |
| Insurance Tier 1 | 2,697 | 8.71% | 0.954 |
| Insurance Tier 2 | 6,030 | 25.01% | — (see `EVAL_RESULTS.md`) |
| Payroll Tier 1 | 44,021 | 50.11% | 0.993 |
| HR (single tier) | 5,364 | 4.77% | 0.774 |
| Financial Services Tier 1 | 4,000 | 7.25% | 0.493 |

Full breakdowns (precision/recall/confusion matrix per layer, plus what
escapes to the LLM tier and why) are regenerated by running each
domain's `eval_harness.py` as a module — see `EVAL_RESULTS.md` in each
domain folder.

RAG quality is separately evaluated via `core/rag/base_rag_evaluator.py`
(custom cosine/LLM-judge locally, RAGAS in production) — one thin
wrapper per domain in `domains/<domain>/.../evaluation/rag_evaluator.py`.

---

## Project structure

```
prism/
├── configs/                    # one YAML per domain — the plug-and-play surface
├── core/
│   ├── config_loader.py        # ConfigLoader — parses + validates domain YAML
│   ├── orchestrator.py         # AgentOrchestrator — instantiates the right Pipeline
│   ├── domain_classifier_agent.py
│   ├── unified_pipeline.py     # UnifiedDomainPipeline — shared fraud+RAG base
│   ├── eval/                   # shared eval-harness engine, all domains
│   └── rag/                    # shared document loader / retriever / evaluator bases
├── domains/
│   ├── banking/                 # fraud/ + documents/, one pipeline.py
│   ├── insurance/                # agents/, evaluation/, pipeline.py
│   ├── payroll_hr/               # payroll/ + hr/, one pipeline.py
│   └── financial_services/       # agents/, evaluation/, pipeline.py
├── data/                        # curated sample documents per domain, for demoing
├── streamlit_app.py             # single UI — classify, confirm, workspace, batch upload
├── main.py                      # CLI smoke test across all 4 domains
├── tests/                       # 192 tests
├── deploy/                      # AWS EC2 provisioning runbook
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Install

```bash
git clone <this repo>
cd prism
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
```

If you're running document chat locally with `LLM_PROVIDER=ollama` (the
default when `ENV=local`), also install `requirements-local.txt` — it adds
`sentence-transformers` (the local CrossEncoder reranker + RAG evaluator),
which pulls in the full CUDA-enabled PyTorch stack and is deliberately
*not* part of the always-installed `requirements.txt` (see that file's
comments — it's dead weight wherever `LLM_PROVIDER=openai` is used,
including the AWS deployment):

```bash
pip install -r requirements-local.txt
```

### 2. Run the CLI smoke test

```bash
python main.py
```

Exercises `AgentOrchestrator` end-to-end for all 4 domains against their
small, committed demo data.

### 3. Launch the UI

```bash
streamlit run streamlit_app.py
```

Upload a file or paste a snippet → Prism guesses the domain → confirm or
override → the matching workspace takes over (fraud verdict + chat).

New here? The landing page has a sample-file picker (e.g. a payslip PDF) —
no upload needed. Try it end to end: **Analyze** → confirm the detected
domain → **Ingest demo document** → read the fraud verdict → ask a question
in the chat panel. A clean demo document returns "0 flagged, 1 clean"; to see
the fraud engine flag something, upload a document with inconsistent figures
(e.g. an inflated gross pay). The classifier's keyword score can be low
(a payslip scores ~6%) — that's expected when the other domains score 0%,
and the LLM tiebreak plus your confirm step settle it.

### 4. Docker Compose

```bash
docker compose up --build
```

Single containerized service on [http://localhost:8501](http://localhost:8501). ChromaDB persists via a mounted volume. For document chat's default `LLM_PROVIDER=ollama`, a running Ollama on the host with `llama3.1:8b` pulled is required — the container reaches it via `host.docker.internal`.

### 5. Tests

```bash
pytest tests/ -v
```

`tests/test_config_loader.py` needs no API key. Most other suites need
`OPENAI_API_KEY`; document-chat suites also need `COHERE_API_KEY` (the
reranker) and, for the default local provider, a running Ollama.

---

## Where the source domains came from

- **Banking** — ported from the standalone [Fraud Detection](https://github.com/Ramya192/fraud-detection) project (fraud side) and [FinLens / Document Intelligence](https://github.com/Ramya192/document-intelligence-system) project (document side): the 4-agent hybrid fraud pipeline with a temporal train/holdout split (no leakage), and the 4-agent RAG pipeline with a dual-backend RAG evaluator, unified under one config id and one pipeline.
- **Insurance, Payroll & HR, Financial Services** — built directly on the same unified-pipeline pattern (`core/unified_pipeline.py`), each with its own tiered ML scorers trained on real public datasets (CMS/SF-Salaries/EMSCAD/blockchain-scam sources — see each domain's `data/prepare_data.py` for provenance).

---

## Author

**Ramya** — Mainframe professional transitioning to Agentic AI.
Target: Agentic AI Developer roles at BFSI firms and Big Tech.
