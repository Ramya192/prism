# domains/

One folder per domain. Everything a domain needs lives inside it, so a new domain
is a new folder here plus one `configs/<id>.yaml`.

```
domains/<domain>/
├── pipeline.py            # thin UnifiedDomainPipeline subclass (the adapter)
├── agents/  tools/        # detector, reasoning and helper code
├── evaluation/            # RAG evaluator wrapper, eval harness
├── data/                  # training + holdout CSVs (git-ignored) and the prepare_*.py scripts
├── reference_corpus/      # the domain's grounding text (Regulation E, CMS, IRS, FinCEN)
└── samples/               # demo documents the UI pickers, tests and main.py use
```

`banking` splits its code into `fraud/` and `documents/`, and `payroll_hr` into
`payroll/` and `hr/`, because each has two fraud/document shapes; their `data/`
folders sit inside those sub-areas, while `samples/` and `reference_corpus/` stay
at the domain level.

## samples/

The UI's "try a sample file" pickers list every `.pdf` / `.docx` directly inside a
domain's `samples/` folder (subfolders are ignored), so anything you don't want in
a dropdown goes in `samples/extra/`.

| Domain | Demo documents (in the pickers) | `samples/extra/` (not in any picker) |
|---|---|---|
| `banking` | `AC00003`, `AC00140`, `AC00254`, `AC00360`, `AC00448` statements (PDF), `transactions_balanced.csv` | `AC00777_statement.{pdf,docx}`, `AC00777_transactions_bulk.csv` |
| `insurance` | `CLM001`, `CLM002`, `CLM003` claims (PDF; CLM003 has approved > billed) | `CLM900_claim.{pdf,docx}`, `CLM900_claims_bulk.csv` |
| `financial_services` | `WALLET001_statement.pdf` | `WALLET900_statement.{pdf,docx}`, `WALLET900_transactions_bulk.csv` |
| `payroll_hr` | `EMP001`, `EMP002`, `EMP003` payslips (PDF; EMP003 has a miscalculated net pay) | `EMP900_payslip.{pdf,docx}`, `EMP900_payroll_register_bulk.csv` |

`payroll_hr/samples/handbooks/` holds the HR handbook test fixture; see the README
there for why the `.docx` itself is not committed.

### Regenerating

- **`extra/` sets** — one PDF, one DOCX and one bulk CSV per domain, with
  deliberately different content from the demo documents (new IDs, different
  transactions and figures). The PDF and DOCX hold the same content (proves both
  loader paths); the CSV is a separate multi-row file in that domain's real Tier 1
  schema (see `core/domain_classifier_agent.py`'s `CSV_SCHEMA_HINTS`).

  ```
  python domains/generate_additional_samples.py
  ```

- **Demo documents** — `insurance/tools/generate_claim_document.py`,
  `financial_services/tools/generate_wallet_statement.py` and
  `payroll_hr/payroll/tools/generate_payroll_register.py` write straight into each
  domain's `samples/`. Banking's `documents/tools/generate_statement.py` writes
  hundreds of statements into a git-ignored folder (`documents/data/statements/`);
  copy the ones you want into `banking/samples/`.
- **`transactions_balanced.csv`** — written by `banking/fraud/data/prepare_data.py`
  (a small 50/50 set sampled from the train split only).
