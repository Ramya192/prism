# ◈ Prism

**Domain-agnostic multi-agent intelligence platform.** The domain is a YAML
config parameter, not a hardcoded assumption — Prism auto-detects what kind
of input it's looking at, a human confirms, and the right multi-agent
pipeline runs. Add a new domain by writing a config file and a thin
pipeline adapter, not by forking the app.

Prism supersedes two earlier standalone portfolio projects — **Fraud
Detection** and **FinLens (Document Intelligence)** — by re-platforming
them as the first two domains on a shared config-driven core, instead of
two separate repos with duplicated orchestration, UI, and deployment
plumbing.

---

## Architecture

```
User uploads document / describes their data
        │
        ▼
┌───────────────────────────┐
│  DomainClassifierAgent    │  keyword overlap against each domain's
│                            │  classification_hints, LLM tiebreak only
│                            │  when the keyword scores are genuinely close
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
│  domains/<id>/pipeline.py  │  domain-specific agent sequence — fraud's
│                            │  detector→analyst→router→alert, or
│                            │  documents' retrieve→reason→validate→eval
└───────────────────────────┘
```

Domains are **not** forced into one shared agent method signature — fraud
analysis and document RAG are genuinely different shapes of work. Every
domain `Pipeline` exposes at least `.run(**kwargs) -> dict`; it's free to
expose more (e.g. `DocumentPipeline.ingest(...)`).

---

## Domains

| Domain | Status | What it does |
|---|---|---|
| `bfsi_fraud` | 🟢 working | Hybrid rule-based → Random Forest → LLM transaction fraud detection. Rules and the ML fast-path resolve most transactions for free; the LLM only reasons over genuinely borderline cases. |
| `bfsi_documents` | 🟢 working | RAG over bank statement PDFs — HyDE + multi-query expansion + RRF fusion + reranking, anomaly detection, and a dual-backend RAG evaluator (custom cosine/LLM-judge locally, RAGAS in production). |
| `hr_compliance` | 🚧 coming soon | Stub — same platform shape as `bfsi_documents`, different corpus (employee handbooks, policy docs). |
| `healthcare` | 🚧 coming soon | Stub — bridges to the standalone [PHI De-identification](https://github.com/Ramya192/phi-deidentification) project's approach (Presidio + spaCy NER, HIPAA Safe Harbor). |

Adding a domain means writing `configs/<id>.yaml` (classification hints +
which agents to wire up) and `domains/<id>/pipeline.py` (the adapter that
actually runs them) — the core (`ConfigLoader`, `AgentOrchestrator`,
`DomainClassifierAgent`, `streamlit_app.py`'s confirmation flow) doesn't
change.

---

## Project structure

```
prism/
├── configs/                    # one YAML per domain — the plug-and-play surface
├── core/
│   ├── config_loader.py        # ConfigLoader — parses + validates domain YAML
│   ├── orchestrator.py         # AgentOrchestrator — instantiates the right Pipeline
│   └── domain_classifier_agent.py
├── domains/
│   ├── bfsi_fraud/              # agents/, tools/, data/, pipeline.py
│   └── bfsi_documents/          # agents/, tools/, evaluation/, data/, pipeline.py
├── streamlit_app.py             # single UI — domain selector + human confirmation
├── main.py                      # CLI smoke test across both working domains
├── tests/
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

### 2. Run the CLI smoke test

```bash
python main.py
```

Exercises `AgentOrchestrator` for both working domains against their small,
committed demo data (a handful of fraud transactions; one committed bank
statement PDF).

### 3. Launch the UI

```bash
streamlit run streamlit_app.py
```

Upload a file or paste a snippet → Prism guesses the domain → confirm or
override → the matching pipeline UI takes over.

### 4. Docker Compose

```bash
docker compose up --build
```

Single containerized service on [http://localhost:8501](http://localhost:8501). ChromaDB persists via a mounted volume. For `bfsi_documents`'s default `LLM_PROVIDER=ollama`, a running Ollama on the host with `llama3.1:8b` pulled is required — the container reaches it via `host.docker.internal`.

### 5. Tests

```bash
pytest tests/ -v
```

`tests/test_config_loader.py` needs no API key. `test_fraud_pipeline.py`
and `test_documents_pipeline.py` need `OPENAI_API_KEY` (and, for the
documents suite's default provider, a running Ollama).

---

## Where the two source domains came from

- **`bfsi_fraud`** — ported from the standalone [Fraud Detection](https://github.com/Ramya192/fraud-detection) project: 4-agent hybrid pipeline, temporal train/holdout split (no leakage), `EVAL_RESULTS.md` regenerated by `domains/bfsi_fraud/eval_harness.py`.
- **`bfsi_documents`** — ported from the standalone [FinLens / Document Intelligence](https://github.com/Ramya192/document-intelligence-system) project: 4-agent RAG pipeline, dual-backend RAG evaluator, and the ATM/Credit transaction-type fix in `tools/generate_statement.py`'s `clean_transactions()` (an ATM-channel transaction is always a withdrawal; the source Kaggle data had some mislabeled `Credit` — fixed before any statement or ground-truth data is generated from it).

Both source pipelines' logic is unchanged in the port — only import paths,
data paths, and (for `bfsi_documents`) the deployment shape (agents called
in-process from one Streamlit app; no separate FastAPI service, since
nothing else consumed that API) moved.

---

## Author

**Ramya** — Mainframe professional transitioning to Agentic AI
Target: Agentic AI Developer roles at BFSI firms and Big Tech
