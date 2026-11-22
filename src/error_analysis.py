"""
Stage 5 — FP/FN error analysis on frozen final test predictions.
Reads:  results/final_test_predictions.csv
        data/raw/credit_default.csv
Writes: results/error_analysis.csv
        figures/error_analysis.png
No tuning, no model selection — diagnostic only.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "final_test_predictions.csv"
RAW = ROOT / "data" / "raw" / "credit_default.csv"
OUT_CSV = ROOT / "results" / "error_analysis.csv"
OUT_FIG = ROOT / "figures" / "error_analysis.png"

# ---- 1. load ----
pred = pd.read_csv(PRED)
raw = pd.read_csv(RAW)
raw = raw.rename(columns={"default payment next month": "default_next_month"})
if "default_next_month" not in raw.columns:
    raw = raw.rename(columns={raw.columns[-1]: "default_next_month"})

# merge on ID (raw has ID)
df = pred.merge(raw, on="ID", how="left", validate="one_to_one")
assert df["default_next_month"].notna().all(), "merge failed"

# ---- 2. choose the frozen model to analyse ----
# Change MODEL only if you decide to analyse a different frozen model.
MODEL = "ensemble_e2"   # options: ensemble_e2, random_forest, gb_balanced, logistic_regression
pred_col = f"pred_{MODEL}"
prob_col = f"prob_{MODEL}"

y = df["y_true"].astype(int).values
yhat = df[pred_col].astype(int).values

tn = int(((y == 0) & (yhat == 0)).sum())
fp = int(((y == 0) & (yhat == 1)).sum())
fn = int(((y == 1) & (yhat == 0)).sum())
tp = int(((y == 1) & (yhat == 1)).sum())
print(f"MODEL={MODEL}  TN={tn} FP={fp} FN={fn} TP={tp}")

# ---- 3. group-wise feature summary ----
df["error_group"] = np.select(
    [(y == 0) & (yhat == 0), (y == 0) & (yhat == 1),
     (y == 1) & (yhat == 0), (y == 1) & (yhat == 1)],
    ["TN", "FP", "FN", "TP"], default="?"
)

key_features = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
key_features = [c for c in key_features if c in df.columns]

summary = df.groupby("error_group")[key_features].agg(["mean", "median"]).round(2)
summary.to_csv(OUT_CSV)
print(f"\nSaved {OUT_CSV}")
print(summary.to_string())

# ---- 4. figure: PAY_0 distribution by error group + prob histogram ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) PAY_0 distribution per group
if "PAY_0" in df.columns:
    sns.countplot(data=df, x="PAY_0", hue="error_group",
                  order=sorted(df["PAY_0"].unique()),
                  ax=axes[0])
    axes[0].set_title(f"{MODEL}: PAY_0 distribution by error group")
    axes[0].set_ylabel("count")

# (b) LIMIT_BAL boxplot per group
if "LIMIT_BAL" in df.columns:
    sns.boxplot(data=df, x="error_group", y="LIMIT_BAL",
                order=["TN", "FP", "FN", "TP"], ax=axes[1])
    axes[1].set_title(f"{MODEL}: LIMIT_BAL by error group")

# (c) predicted probability histogram per true label
sns.histplot(data=df, x=prob_col, hue="y_true", bins=40,
             stat="density", common_norm=False, ax=axes[2])
axes[2].set_title(f"{MODEL}: predicted prob by true label")
axes[2].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
axes[2].legend()

plt.tight_layout()
OUT_FIG.parent.mkdir(exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_FIG}")