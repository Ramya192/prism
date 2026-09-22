# core/unified_pipeline.py
# UnifiedDomainPipeline -- the shared base implementing
# docs/UNIFIED_INGESTION_VISION.md's core rule: any file type (CSV, PDF,
# DOCX) always runs BOTH the domain's fraud verdict and RAG ingestion,
# never gated by which format the upload happens to be. This is not a
# per-domain design option -- it's a standing platform requirement.
#
# What a subclass provides:
#   self.loader / self.retriever / self.reasoning / self.validator --
#     the four RAG agents, same shape every document-capable domain
#     already uses (subclasses of core/rag/base_*).
#   self.detector -- the domain's EXISTING fraud-detection object,
#     already fully built and tested via the CSV path (e.g.
#     HealthcareDetectorAgent, FintechDetectorAgent, or a domain
#     Pipeline like PayrollHrFraudPipeline). Never rebuilt here.
#   EXTRACTION_QUERY -- the fixed internal query the ReasoningAgent uses
#     to pull structured fields out of a PDF/DOCX (mirrors the existing
#     "auto capability" query pattern, e.g. payroll's payroll_audit query).
#   REFERENCE_SOURCE -- core.rag.reference_corpus.REFERENCE_SOURCE_ID if
#     this domain has an ingested reference corpus, else None.
#   DOCUMENT_SCORING_CONTEXT -- optional context string passed to
#     score_record() for PDF/DOCX-derived records, when this domain's
#     detector's LLM tier needs calibration different from CSV-shaped
#     data's own scale (see the class attribute's own docstring).
#   score_record(record) -> dict: calls self.detector with whatever
#     signature it actually has (analyse() returning text to parse, or a
#     pipeline's run() returning a dict already) and normalizes the
#     result to {"predicted": "FRAUD"|"LEGITIMATE", "reason": str}. Public
#     (not a private hook) on purpose: it's also this domain's direct
#     single-record scoring entry point -- e.g. a CSV upload's per-row
#     path, or a Streamlit manual-entry form. Deliberately NOT named
#     run(), which this class reserves for the RAG chat query signature
#     below.
#   _map_extraction_to_records(extraction) -> list[dict]: maps a PDF/
#     DOCX's ReasoningAgent-extracted fields into zero, one, or many
#     records shaped for self.detector -- NOT always one record per
#     document. An Insurance claim document represents one artifact (one
#     record); a Banking statement represents MANY (one per listed
#     transaction) -- "score the artifact, not the entity" means each
#     transaction gets scored on its own, the same as if it had arrived
#     as a CSV row. Returns [] (not a best-effort partial record) when
#     the document genuinely doesn't carry fields the detector's schema
#     needs -- e.g. a bank statement structurally cannot carry Banking
#     Tier1's anonymized PCA features. detect_tier()-style logic inside
#     the detector then correctly falls through to its own LLM-fallback
#     tier for whatever WAS extracted; this method's job is only "can I
#     honestly build recognizable records from what got extracted", not
#     "force a match" or "guarantee exactly one record".
#
# Every ingest() path -- CSV rows, or a PDF/DOCX's extracted records --
# converges on the same result shape: `fraud_verdicts`, a list of zero,
# one, or many verdicts, each independently scored through the SAME
# detect_tier()-routed engine the CSV path already uses. There is no
# separate "single verdict" shape for documents; a document that happens
# to represent exactly one artifact just produces a one-item list.

from __future__ import annotations

import logging
from pathlib import Path

from core.rag.base_document_loader_agent import DuplicateDocumentError
from core.rag.seed_history import build_seed_history

logger = logging.getLogger(__name__)


class UnifiedDomainPipeline:
    EXTRACTION_QUERY: str = (
        "Extract every field on this document relevant to compliance review, "
        "with exact values."
    )
    # Opt-in, not assumed: None until a domain's own pipeline class sets
    # this to reference_corpus.REFERENCE_SOURCE_ID -- and only once that
    # domain has actually ingested a reference corpus under that source
    # id (core/rag/reference_corpus.py's ingest_reference_corpus()).
    # Defaulting this to the real id unconditionally was a real bug: for
    # a domain with no reference corpus (Banking, so far), retrieve()
    # would filter on a source that was never ingested -- harmless when
    # a real document's chunks are also in the filter, but a Chroma
    # get(ids=[]) crash when source_document is None and that
    # nonexistent reference source is the only thing left to search.
    REFERENCE_SOURCE: str | None = None
    MAX_CSV_ROWS_SCORED = 2000  # a demo/interactive-upload ceiling, not a hard platform limit

    # Optional context string passed to score_record() for records mapped
    # from a PDF/DOCX extraction (never for CSV rows, which the caller
    # scores directly and may supply their own context to). None by
    # default. Banking sets this: its Tier 3 LLM prompt/rules were
    # calibrated on card-transaction-scale amounts ($10-$500 "typical"),
    # a real, measured mismatch against bank-statement-scale amounts
    # (salary/rent/ATM withdrawals in the thousands) that caused 100% of
    # a real test statement's transactions to get flagged HIGH -- the LLM
    # had no baseline for what's normal on a STATEMENT specifically, only
    # what's normal on a CARD. This isn't a threshold hack; it's the same
    # `context` parameter FraudDetectorAgent.build_prompt() already takes
    # for exactly this purpose, populated here for document-sourced
    # records specifically.
    DOCUMENT_SCORING_CONTEXT: str | None = None

    # Optional -- a domain with a RAG-quality evaluator (currently only
    # Banking's RAGAS/custom cosine-and-LLM-judge evaluator, live in
    # production) sets self.evaluator in its
    # own __init__; run() calls it if present and folds "evaluation" into
    # the response, exactly reproducing DocumentPipeline.run()'s original
    # try/except behavior. A domain without one (Insurance, so far) never
    # sets it, and the response simply has no "evaluation" key -- not a
    # None value, so existing tests that don't expect the key keep passing.
    evaluator = None

    def score_record(self, record: dict, context: str | None = None) -> dict:
        raise NotImplementedError

    def _map_extraction_to_records(self, extraction) -> list[dict]:
        raise NotImplementedError

    def _extract_document(self, result: dict) -> tuple[str, str | None, list[dict]]:
        """PDF/DOCX extraction, factored out of ingest() so a domain that
        needs to decide WHETHER its extraction query even applies to this
        document (see PayrollHrPipeline's override) can override just
        this hook instead of duplicating ingest()'s CSV/PDF branching and
        seed_history construction. Returns (query_used, extraction_answer,
        records_to_score) -- query_used seeds the opening "user" turn in
        build_seed_history() and defaults to self.EXTRACTION_QUERY here;
        an override that runs a DIFFERENT query (because the fixed
        EXTRACTION_QUERY doesn't apply to this document) returns that
        query instead, so the seed history's opening question actually
        matches what was really asked."""
        chunks = self.retriever.retrieve(self.EXTRACTION_QUERY, source_document=result["document"])
        extraction = self.reasoning.reason(self.EXTRACTION_QUERY, chunks, source_document=result["document"])
        extraction_answer = getattr(extraction, "answer", "")
        records_to_score = self._map_extraction_to_records(extraction)
        return self.EXTRACTION_QUERY, extraction_answer, records_to_score

    def _summarize_verdicts(self, verdicts: list[dict]) -> dict:
        flagged = [v for v in verdicts if v["predicted"] == "FRAUD"]
        return {
            "total_scored": len(verdicts),
            "flagged": len(flagged),
            "clean": len(verdicts) - len(flagged),
        }

    def ingest(self, file_path: str, filename: str | None = None) -> dict:
        """Format decides HOW content is read (loader.load() already
        handles that); it never decides WHETHER a verdict or a chat gets
        built -- both always do. Always produces `fraud_verdicts`, a
        list -- 0, 1, or many entries -- never a format-dependent shape.

        Raises DuplicateDocumentError (before anything is embedded or
        scored) if a file with identical content was already ingested in
        this domain, whatever it was named or whichever batch it came in."""
        existing = self.loader.find_duplicate(file_path)
        if existing is not None:
            raise DuplicateDocumentError(filename or Path(file_path).stem, existing)
        result = self.loader.load(file_path, filename=filename)
        records = result.get("records")

        if records is not None:
            # CSV: every row is already a verdict-ready record.
            query_used = self.EXTRACTION_QUERY
            extraction_answer = None
            records_to_score = records[: self.MAX_CSV_ROWS_SCORED]
        else:
            # PDF/DOCX: one LLM extraction, then zero/one/many mapped
            # records depending on what the document actually represents
            # (see _map_extraction_to_records's docstring and, for when
            # the extraction query doesn't even apply, _extract_document's).
            query_used, extraction_answer, records_to_score = self._extract_document(result)

        if records is not None:
            verdicts = [self.score_record(r) for r in records_to_score]
        else:
            verdicts = [self.score_record(r, context=self.DOCUMENT_SCORING_CONTEXT) for r in records_to_score]
        result["fraud_verdicts"] = verdicts
        result["fraud_summary"] = self._summarize_verdicts(verdicts)

        top_flag = next((v for v in verdicts if v["predicted"] == "FRAUD"), None)
        if verdicts:
            seed_answer = extraction_answer or f"Scored {len(verdicts)} record(s)."
            flag_reason = top_flag["reason"] if top_flag else None
        else:
            seed_answer = extraction_answer or ""
            flag_reason = (
                "No recognizable fraud-schema fields could be extracted from this "
                "document -- verdict unavailable, chat is still available."
            )
        scan_for_seed = {
            "status": "valid",
            "data": {
                "answer": seed_answer,
                "flag": bool(top_flag),
                "flag_reason": flag_reason,
            },
        }

        result["seed_history"] = build_seed_history(query_used, scan_for_seed)
        return result

    def run(
        self, query: str, source_document: str | list[str] | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """Retrieval draws on this document AND the domain's static
        reference corpus (if any) in the same fused ranking -- see
        BaseRetrieverAgent.retrieve()'s `reference_source` param.
        `source_document` also accepts a list of filenames for
        cross-document retrieval -- multiple previously-ingested
        documents compete in that same fused ranking too, not a
        per-document retrieval pass merged after the fact. A single
        string (or None) is unaffected -- passed straight through to
        retrieve()/reason() exactly as before."""
        chunks = self.retriever.retrieve(
            query, source_document=source_document, reference_source=self.REFERENCE_SOURCE,
        )
        output = self.reasoning.reason(query, chunks, source_document=source_document, history=history)
        validation = self.validator.validate(output)

        if validation["status"] == "invalid":
            return {"status": "invalid", "errors": validation["errors"]}

        response = {
            "status": "valid",
            "data": validation["data"].model_dump(),
            "chunks": chunks,
        }

        if self.evaluator is not None:
            try:
                eval_result = self.evaluator.evaluate(
                    query=query, chunks=chunks, answer=output.answer, source_document=source_document,
                )
                response["evaluation"] = eval_result.to_dict()
            except Exception as e:
                logger.warning(f"RAG evaluation skipped: {e}")
                response["evaluation"] = None

        return response
