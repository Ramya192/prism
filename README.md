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

## Status

All four domains — `banking`, `insurance`, `payroll_hr`, `financial_services` — are
working end to end on one shared `UnifiedDomainPipeline`: every upload gets a
fraud verdict and a chat-ready ingestion, each domain has a curated reference
corpus grounding its chat, and each ships a regenerable eval harness.

| | |
|---|---|
| Domains | 4 (banking, insurance, payroll_hr, financial_services) |
| ML tiers | 9 fraud tiers across the 4 domains (HR is single-tier by design) |
| Committed model artifacts | 15 joblib files, ~30 MB (`models/`) — a fresh clone needs no training data |
| Tests | 266 (95 offline tests run in CI; the rest need API keys and/or the train CSVs) |
| Eval | Deterministic-layer F1 per tier against a temporal holdout — see [Evaluation](#evaluation) |

**Known limitations** (deliberate, not oversights):

- **No authentication** on the public demo — only cost guards (rate limits,
  daily upload/chat/scoring budgets, optional access code).
- **CSV scoring ceiling** of 2,000 rows per upload (`PRISM_MAX_CSV_ROWS`), and a
  rolling daily scored-rows budget across all visitors; the UI says when either applies.
- **Drift detection** is built for every domain but wired into live verdicts only
  for Banking; the others measured too noisy (see *What the UI does*).
- **HR** is single-tier (Tier 1 + LLM); no second dataset was found that met the bar.
- **Banking Tier 1** F1 is low (0.234) because fraud is 0.13% of the holdout: recall
  is 0.76 but precision is 0.14 — borderline rows escape to the LLM tier by design.
- **No graph-based fraud-ring detection** (GNN) — most datasets lack the linking
  identifier it needs.
- **Financial Services documents** never carry the account/backend-level fields its
  ML tiers need, so document-sourced records resolve to the LLM tier.

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
| `insurance` | Claim-level fraud (`HealthcareDetectorAgent`): rules → tiered ML (LightGBM Tier 1 / real-world-shaped Tier 2; vetting AUC 0.998 / 0.857 on random splits, temporal-holdout F1s under Evaluation) → LLM claim fraud. | Claim/EOB PDF/DOCX → structured field extraction → same engine. RAG grounded in a CMS Claims Processing Manual (Ch. 26) excerpt. |
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
- **Drift/outlier signal** — an Isolation Forest companion to the
  supervised classifiers asks "does this look like anything in the training
  distribution". It is built for all 4 domains, but only Banking's is wired
  into live verdicts (4.5% legit-outlier rate); the others measured too noisy
  (Payroll 10%, Financial Services 12–14%, HR 43%, Insurance 51%), so they
  ship measured but unwired. See `tests/test_*_drift_detector.py`.
- **Duplicate-upload guard** — a content hash stored in the vector store
  rejects a repeat of an already-ingested document, within a batch and
  across batches. It is scoped to the uploading browser session, and each
  session's documents are stored under their own namespace, so on the shared
  public store one visitor can neither overwrite, block, nor see another's.
- **Upload hardening** — scanned/empty files get a clear message instead of a
  traceback; a CSV past the 2,000-row scoring ceiling says so; rows are
  scored on a small thread pool; uploads are capped at 15 MB.
- **Document-type gate** — every domain checks a PDF/DOCX is actually the kind
  of document its extraction query assumes (payslip, claim, bank/wallet
  statement) before extracting; a wrong-type document degrades to chat-only
  with an honest empty verdict instead of a fabricated record.
- **Cost guards for the public site** — per-session hourly limits and daily
  budgets on uploads, chat, and manual scoring, plus an optional
  `PRISM_ACCESS_CODE` gate (`ui/guard.py`). Because one upload can fan out
  into thousands of LLM calls, scored rows also draw from a rolling daily
  budget (`core/usage_budget.py`, `PRISM_DAILY_SCORED_ROWS_BUDGET`, default
  5,000); past it a document is still ingested and chattable but not
  scored, and the UI says so. It is a cost guard, not authentication; also
  set a spend cap in the OpenAI dashboard.
- **Conversational chat** — multi-turn history, seeded with the fraud
  verdict so a first follow-up like "why was this flagged?" already has it in
  context.
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
| Banking Tier 2 | 3,000 | 29.07% | 0.518 |
| Insurance Tier 1 | 2,697 | 8.71% | 0.954 |
| Insurance Tier 2 | 6,030 | 25.01% | 0.795 |
| Payroll Tier 1 | 44,021 | 50.11% | 0.993 |
| Payroll Tier 2 | 541 | 50.46% | 0.966 |
| HR (single tier) | 5,364 | 4.77% | 0.774 |
| Financial Services Tier 1 | 4,000 | 7.25% | 0.493 |
| Financial Services Tier 2 | 100,000 | 5.95% | 0.470 |

These are the free deterministic layers only. Low F1 on the rare-fraud tiers
(Banking Tier 1 at 0.13%, Financial Services) is mostly a precision effect —
recall stays high (0.76–0.98) at a tiny base rate — and the borderline rows
escape to the LLM tier rather than being forced into a verdict. Full
breakdowns (precision/recall/confusion matrix per layer, plus what escapes to
the LLM tier and why) are regenerated by running each
domain's `eval_harness.py` as a module, which writes an `EVAL_RESULTS.md`
next to it (git-ignored, since it is regenerable; needs the train/holdout CSVs).

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
├── ui/                          # Streamlit UI package (landing, batch upload, shared
│   │                            #   chat/workspace widgets, one screen per domain)
│   └── domains/
├── data/                        # curated sample documents per domain, for demoing
├── docs/                        # UNIFIED_INGESTION_VISION.md, DATA_CONVENTIONS.md
├── streamlit_app.py             # entry point — page setup + step routing only
├── main.py                      # CLI smoke test across all 4 domains
├── tests/                       # 266 tests
├── models/                      # committed joblib artifacts for every ML scorer / drift
│                                #   detector (core/model_store.py) — loaded at startup, no train CSV needed
├── deploy/                      # AWS EC2 provisioning runbook + user-data
├── .github/workflows/ci.yml     # offline tests + Docker build on every push/PR
├── Dockerfile / docker-compose.yml   # python:3.13-slim, single service
└── requirements.txt
```

---

## Quickstart

### 1. Install

```bash
git clone <this repo>
cd prism
# Python 3.13 (matches the Dockerfile)
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

Single containerized service on [http://localhost:8501](http://localhost:8501). ChromaDB persists via the mounted `vector_store/` volume, and every domain's `data/` dir is mounted so ML scorers and sample files aren't stale build-time copies. For document chat's default `LLM_PROVIDER=ollama`, a running Ollama on the host with `llama3.1:8b` pulled is required — the container reaches it via `host.docker.internal`.

**First run only — ingest the reference corpora.** The vector store starts
empty, so chat has no policy grounding until you run, once per fresh store:

```bash
python -m domains.banking.documents.data.ingest_reference_corpus
python -m domains.insurance.data.ingest_reference_corpus
python -m domains.payroll_hr.data.ingest_reference_corpus
python -m domains.financial_services.data.ingest_reference_corpus
```

(In Docker: `docker compose exec prism python -m ...`.) Each is idempotent.

### 5. Trained models

Every ML scorer and drift detector loads a committed artifact from
`models/` at startup (`core/model_store.py`), so a fresh clone — the EC2
box, CI — has working ML tiers without the ~700 MB of gitignored train
CSVs. An artifact is used only if it was built with the same
scikit-learn / LightGBM / XGBoost versions (pinned in `requirements.txt`)
and the train CSV, if present, still matches its hash. After changing the
data, model code, or those pins, regenerate them all. (Every model,
banking's Tier 1 scorer and drift detector included, is built from its
domain's full train set — the same one the Evaluation numbers below use.)

```bash
python -m core.model_store --rebuild    # needs the train CSVs (each domain's prepare_data.py)
```

### 6. Tests

```bash
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the offline, key-free, data-free
suites on every push: config loading, model artifacts, startup wiring,
upload hardening, duplicate guard, rate limiter, conversational RAG. The
rest need `OPENAI_API_KEY`; document-chat suites also need
`COHERE_API_KEY` (the reranker) and, for the default local provider, a
running Ollama; the drift/eval suites need the train/holdout CSVs.

---

## Security notes

- **Prompt injection.** Uploaded documents are untrusted text that reaches
  the LLM. Every reasoning prompt tells the model to treat retrieved context
  as data, not instructions (`core/rag/prompt_safety.py`), and the type gate
  does the same — but that is a mitigation, not a guarantee. The structural
  defences are that the rules/ML tiers score *extracted fields* rather than a
  document's own claims, extraction output is schema-validated, and chat
  output is only ever rendered as text.
- **No authentication.** The public demo is open; see the cost guards above.
- **Model artifacts are pickles** and are loaded only from this repo's own
  `models/` directory, never from user input.

---

## Where the source domains came from

- **Banking** — ported from the standalone [Fraud Detection](https://github.com/Ramya192/fraud-detection) project (fraud side) and [FinLens / Document Intelligence](https://github.com/Ramya192/document-intelligence-system) project (document side): the 4-agent hybrid fraud pipeline with a temporal train/holdout split (no leakage), and the 4-agent RAG pipeline with a dual-backend RAG evaluator, unified under one config id and one pipeline.
- **Insurance, Payroll & HR, Financial Services** — built directly on the same unified-pipeline pattern (`core/unified_pipeline.py`), each with its own tiered ML scorers trained on real public datasets (CMS/SF-Salaries/EMSCAD/blockchain-scam sources — see each domain's `data/prepare_data.py` for provenance).

---

## Author

**Ramya** — Mainframe professional transitioning to Agentic AI.
Target: Agentic AI Developer roles at BFSI firms and Big Tech.
