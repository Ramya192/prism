# Data & ML conventions

Cross-cutting rules for any dataset used to train a scorer in this repo (fraud/anomaly tiers, and anything built the same way later). Not domain-specific — applies to `banking` fraud, `payroll`, and any future domain.

## 1. Currency must be consistent within a domain's fraud/anomaly feature

Every dataset that trains a scorer for the same domain must agree on currency, because rules layers here use fixed, currency-labeled thresholds (e.g. `rule_based_filter()`'s `$0.01`/`$500`/`$1000`), and mixing scales silently breaks them.

- **Identify the source currency before training on any new dataset.** Don't assume — check for a currency column, and if there isn't one, look for other signals (a `Credit_Score` on the 300-850 US FICO scale implies USD; Indian "Designation" terminology plus monthly-figure magnitudes that are only plausible as INR, not USD, implies INR — see `domains/banking/fraud/data/prepare_data.py` and `domains/payroll_hr/payroll/data/prepare_tier2_data.py` for two real examples of inferring this from context, not a label).
- **Convert to one standard currency (USD, so far) using a documented historical rate**, applied once at data-prep time, not at inference time. Every conversion so far cites its exact source and rate:
  - `banking` fraud Tier 1 (`creditcard.csv`, EUR): Fed G.5 release, Sept 2013 average, 1 EUR = 1.3364 USD.
  - `payroll` Tier 2 (`Employee_monthly_salary.csv`, INR): x-rates.com 2020 average, 1 USD = 74.160 INR.
- **Verify the conversion didn't break anything it shouldn't.** A uniform scalar conversion must never change a *ratio*-based feature's meaning (see rule 2) and must never change a *reconciliation identity* — re-derive dependent columns (e.g. `Net_Pay = GROSS - Deduction`) after conversion rather than independently rounding each one, or floating-point rounding can silently break the identity (this happened once — `prepare_tier2_data.py`'s `Net_Pay` is deliberately re-derived rather than converted directly, for exactly this reason).

## 2. Ratio-based features are currency-invariant; absolute-value/peer-comparison features are NOT

This distinction determines whether currency conversion is safe by itself, or actively misleading without more care.

- **Ratio-based checks (safe under conversion, verified empirically):** anything expressed as a fraction of another same-currency figure in the same row — a deduction rate, an overtime-to-base ratio, a reconciliation identity. Multiplying every value in a dataset by the same constant leaves these completely unchanged once `StandardScaler` normalizes by that dataset's own mean/std — confirmed by training the same model on `payroll`'s Tier 2 data in both native INR and converted USD and getting 100% identical predictions, zero score difference.
- **Absolute-value or peer-comparison checks (NOT safe without extra care):** anything that judges a raw figure as "unusual" against a reference population or fixed threshold — e.g. "is this salary high for someone with N years of experience." Currency conversion fixes the *units* of the number but not the *economic context* that makes it rare or normal. A ₹50 lakh (~$60K) salary at 3 years' experience is a genuine outlier against Indian tech-market norms and unremarkable against US tech-market norms — converting the number doesn't tell you which comparison is the right one.
- **Standing rule:** any future feature that compares an absolute figure against a peer group, percentile, or threshold must compute that comparison *within one dataset's own original market context* — never by converting currency and then comparing against a differently-sourced reference population. If a feature like this is ever built, say explicitly which population it's being judged against and why that's a fair comparison.

## Current scorers this applies to

| Domain | Tier | Dataset | Currency handling |
|---|---|---|---|
| `banking` fraud | 1 | `mlg-ulb/creditcardfraud` | EUR→USD converted |
| `banking` fraud | 2 | `mdtalhask/ai-powered-banking-fraud-detection-dataset-2025` | Assumed USD (FICO-scale credit score, no currency field) |
| `payroll` | 1 (ML anomaly) | `kaggle/sf-salaries` | USD (US city government data) |
| `payroll` | 2 | `jb1433/sample-employees-monthly-salary` | INR→USD converted |
| `healthcare` | 1 | `nudratabbas/healthcare-fraud-detection-dataset` | USD (US states/Medicaid, no conversion needed) |
| `healthcare` | 2 | `tejalaveti2306/health-insurance-claims-data-for-fraud-detection` | USD (no conversion needed; unrealistic amount scale flagged separately, not a currency issue) |
| `hr_compliance` | 1 | `shivamb/real-or-fake-fake-jobposting-prediction` | N/A — deliberately excludes salary amounts entirely (multi-country dataset, no currency field, 84% missing); uses presence/absence as a currency-agnostic signal instead |
