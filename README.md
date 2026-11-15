# ML Group Project — V3

6001CMD: Online Shoppers Purchasing Intention (UCI #468). Deadline 22 Nov 2026.

## Team
- M1: EDA + Ensemble
- M2: Shared Preprocessing + Logistic Regression
- M3: Random Forest + Final Evaluation
- M4: Gradient Boosting + Explainability

## Ground rules
- `Revenue` is target (purchase=1); tentative 16 predictors excluding `PageValues` (document rationale).
- Shared 80/20 stratified split with random_state=42; 3-fold shuffled StratifiedKFold with random_state=42.
- Select models by positive-class F1 on TRAINING CV only. Keep final test set untouched until M3 final evaluation.
- Share one set of preprocessing functions from M2; use sklearn Pipeline (fitted within each CV fold).
- Handoffs require real code, numerical evidence and a successful receiver test.

## Getting started (Windows, VS Code terminal)
1. Use Python 3.12 (choose same Python minor version for team). Install VS Code Python and Jupyter extensions.
2. In the repository folder:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -c "import pandas, sklearn; print('OK', pandas.__version__, sklearn.__version__)"
   ```
3. Download `online_shoppers_intention.csv` from https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset and put at `data/raw/online_shoppers_intention.csv`.
4. M1 runs:
   ```powershell
   .\.venv\Scripts\python.exe -m src.data_audit
   ```
   Outputs `results/data_audit.csv` and `results/revenue_distribution.csv`.
5. Once the install and import check work for the whole team, pin packages (one chosen verified environment):
   ```powershell
   .\.venv\Scripts\python.exe -m pip freeze > requirements.txt
   ```
   Commit pinned `requirements.txt`; everyone else should then recreate/sync their environment using `pip install -r requirements.txt`.

## Ownership / handoffs
- M1 -> M2: data audit + findings; M2 -> ALL: preprocessing.py; M2/M3/M4 -> M1: 3 model factories + CV JSONs; M1 -> M3: ensemble.py; M3 -> M4: final metrics/predictions + frozen model; M4 -> all: explanations/report.

Do not upload the university assessment brief to a public repository. Original dataset is CC BY 4.0: attribute its creators and UCI source in the report/README.
