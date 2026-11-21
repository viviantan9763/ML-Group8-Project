
"""
M1 quick data audit for UCI Credit Card Default (#350).
NO preprocessing, feature selection, or model fitting.

Run from repo root:
    python -m src.data_audit
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
XLS = RAW / "default of credit card clients.xls"
CSV = RAW / "credit_default.csv"
OUT = ROOT / "results"

TARGET = "default_next_month"


def main() -> None:
    if not XLS.is_file():
        raise FileNotFoundError(
            f"Missing dataset: {XLS}\n"
            "Download UCI #350 and place the .xls file here."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Read original UCI Excel.
    # The second Excel row contains actual column names.
    data = pd.read_excel(
        XLS,
        sheet_name=0,
        header=1,
        engine="xlrd"
    )

    data.columns = data.columns.map(
        lambda name: str(name).strip()
    )

    # 2. Standardise the target name.
    candidates = [
        "default payment next month",
        "default.payment.next.month",
        "default_next_month"
    ]

    original_target = next(
        (col for col in data.columns
         if col.lower() in candidates),
        None
    )

    if original_target is None:
        raise ValueError(
            "Target column not found.\n"
            f"Available columns: {data.columns.tolist()}"
        )

    data = data.rename(
        columns={original_target: TARGET}
    )

    if "ID" not in data.columns:
        raise ValueError("Expected ID column is missing.")

    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    data[TARGET] = pd.to_numeric(
        data[TARGET], errors="raise"
    ).astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 (No Default) "
            "and 1 (Default)."
        )

    # 3. General data quality audit.
    report = pd.DataFrame({
        "column": data.columns,
        "dtype": data.dtypes.astype(str).to_numpy(),
        "missing_count": data.isna().sum().to_numpy(),
        "missing_percent": (
            data.isna().mean() * 100
        ).round(2).to_numpy(),
        "unique_count": (
            data.nunique(dropna=False).to_numpy()
        )
    })

    report.to_csv(
        OUT / "credit_data_audit.csv",
        index=False
    )

    # 4. Target distribution.
    distribution = (
        data[TARGET]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(TARGET)
        .reset_index(name="count")
    )

    distribution["percent"] = (
        distribution["count"] / len(data) * 100
    ).round(2)

    distribution.to_csv(
        OUT / "credit_target_distribution.csv",
        index=False
    )

    # 5. Categorical code audit.
    categorical_columns = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]

    code_reports = []

    for column in categorical_columns:
        if column not in data.columns:
            raise ValueError(
                f"Expected column missing: {column}"
            )

        counts = (
            data[column]
            .value_counts(dropna=False)
            .rename_axis("code")
            .reset_index(name="count")
        )

        counts.insert(0, "column", column)
        code_reports.append(counts)

    code_audit = pd.concat(
        code_reports, ignore_index=True
    )

    code_audit.to_csv(
        OUT / "credit_categorical_codes.csv",
        index=False
    )

    # 6. Duplicate checks.
    duplicate_ids = data["ID"].duplicated().sum()
    duplicate_full_rows = data.duplicated().sum()

    duplicate_without_id = (
        data.drop(columns=["ID"])
        .duplicated()
        .sum()
    )

    # 7. Create a common CSV format.
    # This is format conversion only.
    # ID and all original predictor values are preserved.
    data.to_csv(CSV, index=False)

    # 8. Display audit summary.
    print("\n=== M1 CREDIT DEFAULT DATA AUDIT ===")
    print("Excel:", XLS)
    print("Rows, columns:", data.shape)
    print("Missing cells:", int(data.isna().sum().sum()))
    print("Duplicate IDs:", int(duplicate_ids))
    print("Duplicate full rows:", int(duplicate_full_rows))
    print("Duplicates excluding ID:", int(duplicate_without_id))

    print("\nTarget distribution:")
    print(distribution.to_string(index=False))

    print("\nColumn types and quality:")
    print(report.to_string(index=False))

    print("\nCategorical code summary:")
    print(code_audit.to_string(index=False))

    print("\nSaved:", CSV)
    print("Saved:", OUT / "credit_data_audit.csv")
    print("Saved:", OUT / "credit_target_distribution.csv")
    print("Saved:", OUT / "credit_categorical_codes.csv")

    print("\nNEXT: Share CSV, audit outputs and notes with M2.")
    print("Do NOT remove duplicates or transform features here.")


if __name__ == "__main__":
    main()
