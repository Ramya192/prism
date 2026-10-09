# ◈ Prism

**Domain-agnostic fraud detection + document intelligence platform.** The
domain is a YAML config parameter, not a hardcoded assumption — Prism
auto-detects what kind of input it's looking at, a human confirms, and the
right multi-agent pipeline runs. Any upload (CSV, PDF, or DOCX), in any
domain, always gets **both** a fraud/anomaly verdict **and** a chat-ready
RAG ingestion — never one or the other based on file format. Add a new
domain by writing a config file and a thin pipeline adapter, not by
forking the app.

---

## Live Demo

**[https://prism-ramya-aws.duckdns.org](https://prism-ramya-aws.duckdns.org)**

Deployed on AWS EC2 (Amazon Linux 2023, `t3.micro`, free tier) as a single
Dockerized Streamlit service. Caddy (a lightweight web server and reverse proxy) is
the HTTPS front door: it serves the app over TLS and issues and renews its Let's Encrypt
certificate automatically. One small instance keeps the deployment simple, reliable and free to
run, and `deploy/README.md` is a step-by-step provisioning runbook for recreating it.

---

## Status

All four domains — `banking`, `insurance`, `payroll_hr`, `financial_services` — are
working end to end on one shared `UnifiedDomainPipeline`: every upload gets a
fraud verdict and a chat-ready ingestion, each domain has a curated reference
corpus grounding its chat, and each ships a regenerable eval harness.

| Domain | Fraud tiers | Documents | Reference corpus (grounds chat) |
|---|---|---|---|
| `banking` | Tier 1 (PCA schema), Tier 2 (real-world schema) | Bank statements | Regulation E (12 CFR 1005) |
| `insurance` | Tier 1, Tier 2 | Claims, EOBs | CMS Claims Processing Manual, Ch. 26 |
| `payroll_hr` | Payroll Tier 1 + 2; HR job postings (single tier) | Payslips | IRS Pub. 15-T |
| `financial_services` | Tier 1 (scam schema), Tier 2 (exchange manipulation) | Wallet / exchange statements | FinCEN CVC guidance |

Across the four: 9 fraud tiers, 15 committed model artifacts (~30 MB, a fresh
clone needs no training data), and 351 tests (179 offline tests run in CI).
Per-tier F1 against a temporal holdout is under [Evaluation](#evaluation).

**Known limitations**

- **No authentication** on the public demo — only cost guards (rate limits, daily
  budgets, optional access code).
- **CSV scoring ceiling** of 2,000 rows per upload; the UI says when it applies.

---

## Architecture

```
Upload (CSV / PDF / DOCX), pasted text, or a batch      ui/landing.py
        │
        ▼
┌──────────────────────────────┐
│  DomainClassifierAgent       │  CSV-header schema match + keyword overlap;
│                              │  LLM tiebreak only when scores are close
└──────────────────────────────┘
        │  best guess + all scores
        ▼
┌──────────────────────────────┐
│  Human confirms or overrides │  ← nothing below runs on the classifier's
│                              │    say-so alone
└──────────────────────────────┘
        │  domain_id
        ▼
┌──────────────────────────────┐
│  ConfigLoader                │  configs/<domain>.yaml → validated DomainConfig
│  AgentOrchestrator           │  builds and caches that domain's pipeline
└──────────────────────────────┘
        │
        ▼
 UnifiedDomainPipeline  (core/unified_pipeline.py, one thin subclass per domain)

 INGEST — every upload                           CHAT — per question
 ─────────────────────────────────────           ─────────────────────────────────────
 1. duplicate check (content hash,               1. rate limit + daily budget
    per browser session)                            (ui/guard.py)
 2. load → chunk → embed → ChromaDB              2. scope gate: is this about the
    (vector_store/)                                 document or the domain?
                                                    (suggested questions skip it)
 3. CSV: each row is a record                    3. retrieve: HyDE + multi-query →
    PDF/DOCX: type gate → retrieve + LLM            RRF fusion over your document AND
    extraction → zero, one or many records          the domain's reference corpus →
    (none → chat-only, no verdict)                  optional Cohere rerank
 4. score each record (thread pool,              4. reason: document text wrapped as
    daily scored-rows budget):                      untrusted data, history included
      Rules → ML tier → LLM                      5. schema-validate the answer
    each verdict records which layer             6. RAGAS panel: faithfulness,
    decided it                                      answer relevancy
 5. seed the chat with the verdict
```

The fraud cascade is cheapest-first: deterministic rules catch the obvious
cases, the ML tier that matches the upload's schema (Tier 1 or Tier 2) decides
whenever it is confident, and only borderline rows — and schemas no ML tier
recognises — reach the LLM. A document with no fraud-scoreable fields (an HR
handbook, a loan agreement) degrades to chat-only, never a fabricated verdict.

**Where retrieval fits.** Retrieval and fraud scoring are separate jobs that share
one retriever:

- **Fraud verdict.** A CSV needs no retrieval: every row is already a record. For a
  PDF/DOCX, retrieval is used once at ingest as a *reader*: it pulls the relevant
  passages out of the document and an LLM extracts structured fields from them
  (amounts, dates, line items). From that point the verdict depends only on those
  fields: rules, then ML, then the LLM tier, which sees the extracted fields plus a
  fixed text note and never any retrieved passage. A document can therefore describe
  itself as legitimate and still be scored on its numbers.
- **Chat.** Retrieval over your document *and* the domain's reference corpus (for
  example Regulation E for Banking) grounds the answers to your questions. Only chat
  uses the reference corpus.

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
`DomainClassifierAgent`, the UI's confirmation flow) doesn't change.

---

## What you can do

- **Upload anything** — CSV, PDF or DOCX in any domain gets both a fraud verdict and
  a chattable document. Several files at once work too, within a domain or
  across domains from the landing page's batch upload.
- **Chat with the result** — a floating, multi-turn assistant (round launcher bubble;
  header bar with maximize/restore and minimize) seeded with the verdict ("why was
  this flagged?"), answering from your document and the domain's reference corpus.
  Each domain offers 2–3 **suggested questions**; every one was run against a real sample
  document and answered correctly on repeat tries. Answers are computed on a background
  thread, so resizing or minimizing the panel mid-answer never loses the question, and
  the conversation, suggestions and input box stay in order (newest at the bottom).
- **See how well it works** — a "How well does it work?" panel on the landing page and in
  each workspace shows the measured F1, holdout size and fraud rate per tier, with the caveats
  (the same figures as [Evaluation](#evaluation)).
- **Know when a document doesn't fit** — a document with nothing to fraud-score says so
  plainly ("No fraud check was run on this document"), never "no anomalies found". If its
  wording clearly belongs to another domain (an insurance claim uploaded into Banking),
  an amber notice points you at the right workspace.
- **Score one record by hand** — manual entry in every domain, plus image input for
  Banking.
- **Read large results at a glance** — a per-layer summary (rules / ML / LLM),
  a collapsed table and a CSV download instead of thousands of rows.
- **Safe on a public URL** — duplicate-upload guard, document-type and scope gates,
  upload caps, per-session rate limits and daily budgets, optional access code
  (see [Security notes](#security-notes)).

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

Isolation Forest drift detectors are built for all four domains, but only Banking's is wired into live verdicts (4.5% legit-outlier rate); the others measured too noisy (Payroll 10%, Financial Services 12–14%, HR 43%, Insurance 51%).

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
├── domains/                     # one self-contained folder per domain — pipeline.py, agents/, tools/,
│                                #   evaluation/, data/ (train CSVs), reference_corpus/, samples/ (demo files);
│                                #   see domains/README.md
│   ├── banking/                 # fraud/ + documents/, one pipeline.py
│   ├── insurance/                # agents/, evaluation/, pipeline.py
│   ├── payroll_hr/               # payroll/ + hr/, one pipeline.py
│   └── financial_services/       # agents/, evaluation/, pipeline.py
├── ui/                          # Streamlit UI package (landing, batch upload, shared
│   │                            #   chat/workspace widgets, one screen per domain)
│   └── domains/
├── docs/                        # UNIFIED_INGESTION_VISION.md, DATA_CONVENTIONS.md
├── streamlit_app.py             # entry point — page setup + step routing only
├── main.py                      # CLI smoke test across all 4 domains
├── tests/                       # 351 tests
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

Upload a file or paste a snippet → Prism guesses the domain → the guess appears
beside the upload, with the pipeline diagram and a confirm/override box, so nothing
needs scrolling → confirm → the matching workspace takes over (fraud verdict + chat).
The PRISM header stays pinned in the top bar throughout.

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
- **Uploads on the public demo: what happens to them.** The demo has no sign-in, so it is
  built to keep as little as possible:
  - The file is written to a temporary file that is deleted as soon as ingestion finishes.
    What the app keeps is the extracted text, split into chunks with its embeddings, in the
    vector store, tagged with your browser-session ID. Other visitors cannot reach it through
    the app.
  - To read, answer and score, text from your document is sent to OpenAI's API (extraction,
    embeddings, chat and the LLM scoring tier), photos go to GPT-4o Vision, and, when an optional
    Cohere API key is configured, the retrieved passages go to Cohere's reranking service to be
    ordered by relevance. The fraud rules and ML models run inside the app's own server with
    no external call; only borderline rows reach the LLM.
  - A scheduled job (`core/rag/purge_old_uploads.py`, set up in `deploy/README.md`) deletes
    uploads older than 24 hours.
  - Because there is no authentication, whoever administers the server could read stored
    chunks while they exist.
- **Model artifacts.** Trained models load only from this repository's own `models/`
  directory, never from user input, and only when their recorded library versions (and
  training-data hash, when the data is present) match.

---

## Future work

- **Image input for the other domains.** Banking can already score a receipt or cheque
  photo (GPT-4o Vision reads the amount); a payslip photo, a claim form or a wallet
  screenshot would extend the same path to Payroll & HR, Insurance and Financial Services.
- **Fraud-ring detection.** Graph-based detection of linked accounts, once datasets carry a
  linking identifier (most of the ones used here do not).
- **Sign-in for the public demo**, replacing the cost guards and optional access code.
- **Drift detection beyond Banking.** The Isolation Forest detectors exist for all four
  domains; wiring the others into live verdicts waits on their outlier noise rates coming down.
