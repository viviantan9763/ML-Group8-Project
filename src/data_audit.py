"""M1 quick data audit; NO preprocessing or model fitting.

Run from repo root: python -m src.data_audit
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'raw' / 'online_shoppers_intention.csv'
OUT = ROOT / 'results'


def main() -> None:
    if not CSV.is_file():
        raise FileNotFoundError(
            f'Missing CSV: {CSV}\nDownload the UCI #468 CSV and place it at this path.'
        )

    data = pd.read_csv(CSV)
    if 'Revenue' not in data.columns:
        raise ValueError("Target column 'Revenue' missing. Check that you downloaded UCI #468.")

    OUT.mkdir(exist_ok=True)
    report = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes.astype(str).to_numpy(),
        'missing_count': data.isna().sum().to_numpy(),
        'missing_percent': (data.isna().mean() * 100).round(2).to_numpy(),
        'unique_count': data.nunique(dropna=False).to_numpy(),
    })
    report.to_csv(OUT / 'data_audit.csv', index=False)

    revenue = (data['Revenue'].astype(str)
               .value_counts(dropna=False)
               .rename_axis('Revenue')
               .reset_index(name='count'))
    revenue['percent'] = (revenue['count'] / len(data) * 100).round(2)
    revenue.to_csv(OUT / 'revenue_distribution.csv', index=False)

    print('=== M1 DATA AUDIT ===')
    print('CSV:', CSV)
    print('Rows, columns:', data.shape)
    print('Duplicate rows:', data.duplicated().sum())
    print('Missing cells:', int(data.isna().sum().sum()))
    print('\nTarget distribution:\n', revenue.to_string(index=False))
    print('\nColumn types:\n', report[['column','dtype','missing_count','unique_count']].to_string(index=False))
    print('\nSaved:', OUT / 'data_audit.csv')
    print('Saved:', OUT / 'revenue_distribution.csv')
    print('NEXT: Share these outputs and notes with M2. DO NOT delete duplicates or scale data here.')


if __name__ == '__main__':
    main()
