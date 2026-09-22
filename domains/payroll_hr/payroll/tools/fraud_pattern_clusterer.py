# domains/payroll_hr/payroll/tools/fraud_pattern_clusterer.py
# HDBSCAN -- an UNSUPERVISED analysis run AFTER classification, on
# records a scorer has already flagged as fraud, to reveal whether
# "fraud" in this domain is one pattern or several distinct sub-types.
# Not a detection layer itself -- a post-hoc analytics tool for a human
# reviewer, and a validation of the labeling scheme's own structure.
#
# VALIDATED on payroll Tier 1's flagged training rows: blindly (no label
# used) recovers the exact 3 injected redistribution patterns from
# data/prepare_tier_ml_data.py -- overtime-heavy, other-heavy, and
# base-only splits -- as 3 clean, near-equal-sized, ZERO-noise clusters,
# when run on Base/OT/Other as a FRACTION of total pay rather than raw
# dollar amounts. Raw dollar amounts over-segment into ~50 small
# clusters, because salary SCALE varies a lot across pay levels within
# each pattern type -- HDBSCAN's density-based approach treats that
# scale variation as separate neighborhoods. Ratios remove that confound
# since they're already on a comparable 0-1 scale regardless of salary
# level.

import pandas as pd
from sklearn.cluster import HDBSCAN


class FraudPatternClusterer:
    RATIO_COLS = ["base_pct", "ot_pct", "other_pct"]

    def __init__(self, min_cluster_size: int = 200, min_samples: int = 20):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    @staticmethod
    def _add_ratios(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["base_pct"] = df["BasePay"] / df["Stated_TotalPay"]
        df["ot_pct"] = df["OvertimePay"] / df["Stated_TotalPay"]
        df["other_pct"] = df["OtherPay"] / df["Stated_TotalPay"]
        return df

    def cluster(self, flagged_df: pd.DataFrame) -> pd.DataFrame:
        """Takes rows already flagged as fraud (by whatever scorer) and
        returns them with a `cluster` column added (-1 = doesn't fit any
        dense cluster -- a genuinely ambiguous/borderline case, not an
        error)."""
        df = self._add_ratios(flagged_df)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        df["cluster"] = clusterer.fit_predict(df[self.RATIO_COLS])
        return df

    def summarize(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """One row per cluster: size and its characteristic Base/OT/Other
        split -- what a human reviewer would actually want to see."""
        rows = []
        for c in sorted(clustered_df["cluster"].unique()):
            sub = clustered_df[clustered_df["cluster"] == c]
            rows.append({
                "cluster": c,
                "size": len(sub),
                "base_pct": sub["base_pct"].mean(),
                "ot_pct": sub["ot_pct"].mean(),
                "other_pct": sub["other_pct"].mean(),
            })
        return pd.DataFrame(rows)
