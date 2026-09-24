# Unified ingestion — target architecture (mostly built)

This is the standing spec for where Prism's upload pipeline is headed, and how much of it is actually built. Written once so it doesn't need re-explaining — read this before touching ingestion, classification, or the fraud/RAG split.

## The target, in one sentence

**Any domain accepts any file type (PDF, CSV, DOCX). Once the domain is confirmed, the system automatically runs BOTH the fraud/anomaly verdict AND RAG ingestion on the same upload — never one or the other based on file format.**

This is now true for all 4 business domains (`banking`, `payroll_hr`, `insurance`, `financial_services`), each on one unified pipeline (`core/unified_pipeline.py`'s `UnifiedDomainPipeline`). File type never gates which capability runs — fraud/anomaly scoring and RAG ingestion are standing requirements for every upload, not a choice made per format.

## What "unified" means, concretely (now built, all 4 domains)

1. **Extraction is a normalization step, not a separate detection mechanism.** Each domain's `ReasoningAgent` extracts structured fields from a PDF/DOCX via LLM. `_map_extraction_to_records()` maps those fields into the exact record shape the domain's CSV-based detector already expects — returning zero, one, or many records depending on what the document actually represents ("score the artifact, not the entity": one record for a claim/payslip, many for a bank/wallet statement's transactions).
2. **That reshaped record runs through the SAME rules → ML → LLM pipeline** the CSV path already uses, via `score_record()`.
3. **RAG chat runs independently, in parallel** on the same upload — extraction feeding the fraud verdict doesn't replace the document being chunked/embedded/retrievable for open Q&A.
4. **A document with no fraud-scoreable fields degrades gracefully, not incorrectly** — `_map_extraction_to_records()` returning `[]` produces `fraud_verdicts: []` and a clear `flag_reason` ("no recognizable fraud-schema fields... chat is still available"), not a fabricated record. This is what makes an HR handbook upload (see below) work without any content-type branching in the pipeline itself.

## Any file format (built)

`core/rag/base_document_loader_agent.py`'s `load()` dispatches by extension: `.csv` (rows become both chattable chunks and `records` for direct scoring), `.docx` (via `python-docx`), `.txt`/`.md` (reference corpora), else PDF (via `pdfplumber`).

CSV-header content-sniffing into `DomainClassifierAgent` is now built: a raw CSV column dump routes to the right domain the same way free text does, via `CSV_SCHEMA_HINTS` in `core/domain_classifier_agent.py` — the classifier's first line is checked for a CSV-header shape, then matched against each domain's known Tier1/Tier2 column schemas (the same `REQUIRED_COLS`/`FEATURE_COLS` each domain's own scorer already checks). A match covering ≥66% of a known schema is decisive (`method="csv_schema"`); anything else (a CSV shaped like nothing Prism has trained on) falls through to the existing keyword/LLM path unchanged. Covered by `TestCsvSchemaClassification` in `tests/test_config_loader.py` (6 cases: one per known Tier1 schema across all 4 domains, plus an unrecognized-header fallback and a comma-in-prose negative case). Not covered: a CSV whose columns don't match any known schema at all — those still rely on keyword matching over the decoded header text.

## Chat draws on more than the uploaded document (built, all 4 domains)

`BaseRetrieverAgent.retrieve()`'s `reference_source` param merges a domain's static reference corpus into the SAME HyDE/multi-query/RRF/rerank ranking as the user's own document chunks — not a second retrieval pass merged after the fact. Real, fetched-and-curated reference corpora, each ingested via `domains/<domain>/.../data/ingest_reference_corpus.py` and verified present in the vector store:

- **`insurance`**: [CMS Medicare Claims Processing Manual, Chapter 26](https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c26.pdf) — claim validity/compliance rules (10.1 incomplete claims, 10.2 required fields, MSP circumstances). `domains/insurance/reference_corpus/cms_claims_processing_excerpt.txt`. E2e-verified.
- **`payroll_hr`**: [IRS Publication 15-T (2026)](https://www.irs.gov/pub/irs-pdf/p15t.pdf) — Percentage Method / Wage Bracket Method withholding tables, real 2026 dollar brackets, Form W-4 step effects, rounding rules. `domains/payroll_hr/reference_corpus/irs_pub15t_excerpt.txt`. **E2e-verified** (`tests/test_payroll_pipeline.py::test_followup_query_draws_on_reference_corpus`, real quoted $53 Weekly Wage Bracket figure).
- **`banking`**: [Regulation E, 12 CFR Part 1005](https://www.consumerfinance.gov/rules-policy/regulations/1005/) — unauthorized-EFT definition (§1005.2(m)), consumer liability tiers (§1005.6, $50/$500/60-day), error-resolution procedures and timelines (§1005.11). `domains/banking/reference_corpus/regulation_e_excerpt.txt`. **E2e-verified** (`tests/test_documents_pipeline.py::TestReferenceCorpus`, real quoted $50 liability-cap figure).
- **`financial_services`**: [FinCEN Guidance FIN-2019-G001](https://www.fincen.gov/sites/default/files/2019-05/FinCEN%20Guidance%20CVC%20FINAL%20508.pdf) — CVC/money-transmitter definitions, MSB AML program requirements, P2P exchangers, hosted vs. unhosted wallets. `domains/financial_services/reference_corpus/fincen_cvc_guidance_excerpt.txt`. **E2e-verified** (`tests/test_financial_services_pipeline.py::test_followup_chat_draws_on_reference_corpus`, real hosted/unhosted wallet distinction).

All 4 domains' reference corpora are now e2e-verified — confirmed present in each domain's vector collection directly, and confirmed actually grounding chat answers (not just retrievable) via a real question each corpus alone can answer correctly.

## HR handbook RAG (built — required a real content-classification gate, not graceful degradation alone)

A handbook is not a fraud-scoreable artifact. The first attempt at this assumed `payroll_hr`'s pipeline needed **zero new content-type routing** to handle one — that `_map_extraction_to_records()` returning `[]` (the base class's documented behavior for a document with no fraud-schema fields) would be enough. **Running the real e2e test against an actual SHRM employee handbook proved that assumption wrong**: asked `EXTRACTION_QUERY`'s payslip-specific question, the LLM didn't reliably return nulls — it fabricated a plausible-looking `GROSS`/`Deduction`/`Net_Pay` record with dollar figures that appear *nowhere* in the real handbook text (verified directly against the source), which `_map_extraction_to_records()` then honestly turned into a real FRAUD verdict on a document that was never a payslip.

The actual fix: `PayrollHrPipeline._looks_like_payslip()` classifies the document (a cheap, unstructured LLM call — deliberately not `self.reasoning.reason()`, which would force-fit the payslip schema onto whatever it's given) *before* `EXTRACTION_QUERY` ever runs. A non-payslip document never reaches a prompt that pressures the LLM into inventing payslip-shaped numbers — it gets a plain-text summary and an honestly-empty record list instead. This required generalizing `UnifiedDomainPipeline` with a new `_extract_document()` hook (factored out of `ingest()`) that any future domain needing the same kind of pre-extraction gate can reuse.

Real fixture: SHRM's actual "Sample Employee Handbook (2023)" template (`domains/payroll_hr/data/handbooks/` — not committed, SHRM-members-only redistribution terms, see that directory's `README.md` for the re-download command). **E2e-verified**: `tests/test_payroll_hr_handbook_pipeline.py` (4/4) plus a full re-run of the existing payslip suite (`tests/test_payroll_pipeline.py`, 7/7) to confirm the new classification gate causes zero regression on real payslips.

## Multi-document / cross-document retrieval (built)

`source_document` now accepts a `list[str]` everywhere it flowed through as a single filename: `BaseRetrieverAgent.retrieve()`, `UnifiedDomainPipeline.run()`, and every domain's `ReasoningAgent.reason()`/`_build_prompt()`. Multiple ingested documents compete in the same fused HyDE/variant/RRF/rerank ranking as a document + its domain's reference corpus already did — genuinely the same mechanism, generalized, not a second code path. A plain string (or `None`) is completely unaffected — exact prior single-document behavior, unchanged. **E2e-verified** (`tests/test_cross_document_retrieval.py`, 3/3): a single query spanning Insurance's two existing demo claims correctly stated both documents' real dollar figures, and direct retrieval confirmed chunks returned from both sources.

Multi-file **upload** at the workspace level is now built too: `render_document_upload()` in `streamlit_app.py` (shared by Insurance/Financial Services/Payroll & HR) uses `st.file_uploader(..., accept_multiple_files=True)` — several files can be selected and ingested in one batch, each independently scored (its own fraud verdict, shown in its own expander), and all of them become jointly chattable through the cross-document retrieval above (`source_document` accumulates as a list). Verified interactively in-browser: the widget accepts multiple files, the ingest button correctly pluralizes ("Ingest N uploaded files"), and ingestion runs per-file without error. What this does NOT cover: the top-level "what are you checking today?" drop zone is still single-file/single-domain — batch-uploading files that belong to *different* domains, each independently classified, is a bigger re-architecture of the classify-then-confirm flow, deferred by choice, not an oversight (see the top-level batch-upload section below, which covers that case separately).

Streamlit UI for Insurance, Payroll & HR (fraud side), and Financial Services is built — all 3 share `render_workspace()`/`render_document_upload()`/`render_chat_panel()` (upload + manual-entry tabs, a docked chat panel, per-domain manual-entry forms). Only Banking keeps its own richer bespoke 4-tab UI (transaction tables, RAG-eval metrics inline) — a deliberate exception, not a gap.

A RAG-quality evaluator exists for all 4 domains (`domains/<domain>/.../evaluation/rag_evaluator.py`, all thin wrappers over the shared `core/rag/base_rag_evaluator.py`).

## CSV-header content-sniffing (built)

`DomainClassifierAgent` (`core/domain_classifier_agent.py`) now matches a raw CSV header row against `CSV_SCHEMA_HINTS` — each domain's known Tier1/Tier2 column schemas (the same `REQUIRED_COLS`/`FEATURE_COLS` each scorer already checks) — before falling back to keyword matching. A ≥66%-of-schema match is decisive (`method="csv_schema"`); anything else falls through to the keyword/LLM path unchanged. Covers all 4 domains' known schemas; a CSV shaped like nothing Prism has trained on still relies on keyword matching over the decoded text, same as before. Tested: `TestCsvSchemaClassification` in `tests/test_config_loader.py` (6 cases).

## Ground-truth eval harnesses beyond raw-classifier calibration (built, all tiers, all domains)

Every domain's `evaluate.py`/`prepare_data.py` already answers "how good is the raw classifier" (PR-AUC/ROC-AUC/threshold sweeps) — that's classifier calibration. A separate, harder question — "how good is the actual deployed rules→ML→LLM system a user would hit" — is now answered for every domain, not just Banking's Tier 1 (`domains/banking/fraud/eval_harness.py`, the original implementation). Generalized via a shared engine, `core/eval/base_fraud_eval_harness.py` (every domain's `DetectorAgent` had independently converged on the same `rule_based_filter()`/`ml_filter()` shape, so this reuses banking's original logic rather than re-deriving it per domain):

- `domains/banking/fraud/eval_harness.py` — Tier 1 unchanged (existing, tested logic untouched); Tier 2 added via the shared engine, appending its own section to the same `EVAL_RESULTS.md`.
- `domains/insurance/eval_harness.py`, `domains/financial_services/eval_harness.py`, `domains/payroll_hr/payroll/eval_harness.py` — new, both tiers each.
- `domains/payroll_hr/hr/eval_harness.py` — new, single tier (HR stays single-tier permanently, by design).

Run as a module, e.g. `python -m domains.insurance.eval_harness` (writes `EVAL_RESULTS.md` in that domain's folder). All 5 scripts run clean end-to-end against their real holdout data.

**Update: the LLM-sample extension (below) is now built for all 5 scripts, not just Banking's.**

## LLM-sample extension for every eval harness (built)

Banking's `eval_harness.py` pioneered a small, honestly-labeled stratified real-LLM sample (`--with-llm-sample`) on top of the free deterministic-layer numbers above — reported separately, never folded into the regression-tested headline numbers, since it's a small sample whose exact composition depends on the random seed. Generalized via `LlmSampleSpec`/`run_llm_sample_eval`/`render_llm_sample_markdown` in `core/eval/base_fraud_eval_harness.py` (every domain's `DetectorAgent` had independently converged on the same `rule_based_filter()`/`ml_filter()`/`build_prompt()`/`parse_response()`/`is_fraud()` shape, reused rather than re-derived per domain). Every domain's `eval_harness.py` now accepts `--with-llm-sample`:

- `domains/banking/fraud/eval_harness.py` — unchanged, its own hand-written original implementation (not migrated to the shared engine, to avoid touching working, already-verified logic).
- `domains/insurance/eval_harness.py`, `domains/financial_services/eval_harness.py`, `domains/payroll_hr/payroll/eval_harness.py` — both tiers each, via the shared engine.
- `domains/payroll_hr/hr/eval_harness.py` — single tier.

Real API calls cost money and take real wall time (one call per record that escapes rules+ML, ~150 records × up to 2 tiers per domain) — verified correct via a mocked-LLM smoke test across all 4 non-banking domains (functools.partial tier binding, bare binding for HR, scorer.score(), markdown rendering) rather than by actually spending API credits during development.

## Top-level batch upload across different domains (built)

`streamlit_app.py`'s landing page now has a second upload path alongside the single-file/single-domain command bar: a "Batch upload" expander accepting multiple files at once, each independently classified (CSV: real content; PDF/DOCX: filename-only, same honesty as the single-file flow), staged with an editable per-file domain override, and — only after one combined human confirm click, same trust rule as the single-file flow — routed through `AgentOrchestrator.get_pipeline(domain_id).ingest()` for its *own* confirmed domain. Results are grouped by domain on a dedicated results screen (`render_batch_results_view`), each showing its fraud verdict via the same `render_fraud_verdicts` the single-domain workspace uses, with an "Open `<domain>` workspace to chat" button per domain that seeds that domain's normal per-domain session state (`doc_key`/`verdicts_key`/`history_key`) so chat continues in the existing workspace UI rather than a third, parallel one. Banking's own tab4 uses a different session-state contract (tracks one `selected_doc`, not a list) — only the last-ingested Banking file in a batch carries into its chat, every file's verdict still shown regardless. Verified end-to-end with Playwright against a live-running instance: stage → override → confirm → real ingest → grouped results → workspace handoff, including catching and fixing a real bug where the handoff button didn't navigate (the results-view routing condition didn't yet account for `confirmed_domain` and `batch_results` both being set at once).

## What's still open

Nothing outstanding right now. (Update 2026-09-24: the duplicate check is now scoped to the uploading browser session — `find_duplicate(file, owner)` — and each session's documents are stored as `<filename>@<owner>`, so on the shared public store one visitor can't overwrite, block, or see another's uploads. Same pass: scanned/empty files raise `EmptyDocumentError`, a CSV past the scoring ceiling reports `truncated`, and every domain has the document-type gate that `payroll_hr` originally introduced for handbooks.) The one item this section used to list — no cross-*batch* dedup/merge — is resolved: `BaseDocumentLoaderAgent.find_duplicate()` (`core/rag/base_document_loader_agent.py`) keys on a content hash stored in the vector store itself, not session state, so it catches a repeat upload across batches and across sessions for as long as the vector store holds that document. Covered by `tests/test_duplicate_ingest.py` and exercised through the batch-upload path via `DuplicateDocumentError` in `streamlit_app.py`'s `render_batch_upload_section`.
