# HR handbook test fixture

`tests/test_payroll_hr_handbook_pipeline.py` exercises `PayrollHrPipeline`'s
document path against a real employee handbook — not a payslip — to prove
that a genuinely different HR document type still gets a graceful
`fraud_verdicts: []` (not a crash, not a forced payslip-shaped guess) and
still gets full RAG chat.

The fixture itself is **not committed** (see `.gitignore`): it's SHRM's
real "Sample Employee Handbook (2023)" template, whose own first paragraph
states it "is available only to SHRM members and may be downloaded and
modified for your organization" — redistributing the file itself inside
this public repo isn't consistent with that term, even though using it
locally as a private test fixture is exactly the kind of use it's meant for.

To run that test locally:

```
curl -o domains/payroll_hr/data/handbooks/SHRM_Sample_Employee_Handbook_2023.docx \
  https://www.shrm.org/content/dam/en/shrm/business-solutions/SHRM-Sample-Employee-Handbook-2023.docx
```

The test is `skipif`-guarded on this file's presence, same pattern as
`domains/insurance/data/statements/CLM001_claim.pdf` and
`domains/financial_services/data/statements/WALLET001_statement.pdf` (both
of those are self-generated synthetic demo data, not third-party content,
so they ARE committed — this is the one exception in the project).
