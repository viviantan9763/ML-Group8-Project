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

## Random forest experiment

The current implementation uses the checked-in **UCI Credit Card Default** data and the [shared preprocessing contract](results/preprocessing_contract.md). Run from the repository root after installing the requirements.

```powershell
.\.venv\Scripts\python.exe -m src.random_forest
```

The pipeline one-hot encodes the categorical predictors and passes numeric predictors through. It fits preprocessing within each of the three training folds. The experiment compares a default forest with 48 randomly sampled configurations on the same folds, selects by mean positive-class F1, and refits the winner on all 24,000 training rows. The held-out test split is reserved for final evaluation.

- `results/random_forest_cv.json` records the baseline and tuned fold scores, their mean and standard deviation, and the selected parameters.
- `results/random_forest_search.csv` ranks all sampled settings by mean F1 and includes secondary metrics and fit times.
- `artifacts/random_forest.joblib` holds the fitted preprocessing and classifier. Pass the original predictor columns from `split_data()` when making predictions.
- `src.random_forest.make_random_forest(**params)` builds a fresh pipeline for the ensemble. Pass the `selected_params` object from the CV summary.

For a quick smoke run, use `--n-iter 2 --output-dir results/rf_smoke --model-path artifacts/rf_smoke.joblib` to keep its outputs separate. Use `--jobs 1` for serial execution or `--jobs -1` to allow all available cores.

### Tuning evidence GUI

For the full Streamlit studio with configurable baseline and search, evaluation dashboards, comparison, history, and exports, see the [Random Forest Builder guide](m3_workspace/README.md).

```powershell
.\.venv\Scripts\python.exe -m src.random_forest_gui
```

The GUI opens the saved CV evidence from `results`. Choose **Replay saved scores** to animate the original draw order and see how the best F1 improved. Select a ranked row to inspect all seven settings and its three fold scores. **Search space** lists the values available to the sampler and compares baseline and tuned CV metrics.

Choose **Run search** to train again and watch fold scores arrive in the live fit log. The default is 48 candidates. Use 2 candidates for a shorter live demonstration. New runs save their CSV, JSON, and fitted pipeline in a timestamped folder under `results/rf_gui_runs`.

Do not upload the university assessment brief to a public repository. Original dataset is CC BY 4.0: attribute its creators and UCI source in the report/README.
