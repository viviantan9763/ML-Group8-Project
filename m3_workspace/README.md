# Random Forest Builder

This Streamlit app configures and runs Random Forest experiments for the Credit Card Default project. The official dataset uses M2's shared 80/20 stratified split with seed 42 and shared one-hot preprocessing inside each training fold. Sandbox runs use seeded synthetic data and appear in a separate history. Neither mode scores its holdout during training or tuning.

## Windows and PyCharm

From the repository root, using an installed Python 3.12:

```powershell
py -3.12 -m venv m3_workspace/.venv
.\m3_workspace\.venv\Scripts\python.exe -m pip install -r m3_workspace/requirements.txt
<<<<<<< Updated upstream
.\m3_workspace\.venv\Scripts\python.exe -m streamlit run m3_workspace/app/main.py
```

In PyCharm, select `m3_workspace\.venv\Scripts\python.exe` as the project interpreter. Add a **Python** run configuration with **Module name** set to `streamlit`, **Parameters** set to `run m3_workspace/app/main.py`, and the repository root as **Working directory**. Use the local URL printed in the Run console.
=======
.\.venv\Scripts\python.exe m3_workspace/app/main.py
```

Running `.\.venv\Scripts\python.exe m3_workspace/app/main.py` directly (including PyCharm's normal Run button) auto-boots `streamlit run` and opens your browser automatically, or you can run `python -m streamlit run m3_workspace/app/main.py` manually. In PyCharm, select `.venv\Scripts\python.exe` (or `m3_workspace\.venv\Scripts\python.exe`) as the project interpreter with the repository root as **Working directory**.
>>>>>>> Stashed changes

## Suggested demonstration

1. Select **Synthetic sandbox** for a quick demonstration, or **Official credit data** for assessment experiments.
2. In **Cross-Validation**, use 3 folds, shuffle, CV seed 42, and F1 for the group protocol.
<<<<<<< Updated upstream
3. In **Baseline Builder**, load the seeded group baseline and click **Run baseline**. Configure tree settings in the baseline and advanced panels. The scikit-learn reset sets the library's defaults, including a blank seed (`null`). Enter an integer seed before starting a reproducible experiment.
4. In **Hyperparameter Tuning**, choose **Quick Test**, **Balanced Search**, **Thorough Search**, or edit **Custom Search**. Presets fill in explicit value lists. Check the candidate count and fit budget, then click **Run tuning**. The engine filters bootstrap combinations before counting candidates.
=======
3. In **Baseline Builder**, load the seeded group baseline (or click **Load KB calibrated preset**) and click **Run baseline now & show graphs**. Configure tree settings in the baseline and advanced panels. The scikit-learn reset sets the library's defaults, including a blank seed (`null`). Enter an integer seed before starting a reproducible experiment.
4. In **Hyperparameter Tuning**, first expand **OOB `n_estimators` convergence check (`[25..600]`)** to size `n_estimators` from the 2×2 zoomed OOB plateau curves, then choose **Quick Test**, **KB Focused Search** (default), **Balanced Search**, **Thorough Search**, or edit **Custom Search**. Presets fill in explicit value lists. Check the candidate count and fit budget, then click **Run tuning now & show graphs** (or queue in background). The engine filters bootstrap combinations before counting candidates.
>>>>>>> Stashed changes
5. In **Export & History**, watch the worker log and use **Refresh completed results** after completion. Tuning log verbosity 2 or higher shows completed fit messages.
6. In **Results & Evaluation**, inspect mean and fold-level scores, the candidate ranking, and the best score by draw. The tuned model's OOF estimates reuse the tuning folds and can be optimistic.
7. In **Experiment Comparison**, select the baseline and tuned run made with matching dataset, split, pipeline and CV partitions. Export the comparison CSV or download the run's candidate CSV, configuration, parameters, summary, PNG, and trained model from **Export & History**.
8. In **Random Forest Visualizer**, open a fitted run, then replay actual bootstrap draws or tree votes, inspect a decision path, or compare measured complexity and performance across compatible runs.

Runs are persisted under `m3_workspace/runs/<dataset>/<run_id>/`. One worker can run at a time across browser sessions. Runs save the requested configuration, effective estimator parameters, versions, dataset and split fingerprints, fold scores, timings, plots data and model. A worker that fails records its error. The log remains available for diagnosis.

Run an exported configuration from the CLI with:

```powershell
.\m3_workspace\.venv\Scripts\python.exe -m m3_workspace.engine.experiment_manager --config path/to/config.json
.\m3_workspace\.venv\Scripts\python.exe -m pytest -q m3_workspace/tests
```

## Group-dependent actions

<<<<<<< Updated upstream
- **M1 handoff.** The exact `results/rf_cv.json` schema is absent from the checkout. Standard evidence exports are available. The Studio does not generate or overwrite the official handoff file while the schema remains unknown.
=======
- **M1 handoff.** Tab 8 (**Export & History**) generates and exports both `m3_workspace/exports/rf_ensemble_handoff.json` (full KB audit, `selected_params`, and OOF threshold recommendation) and the M1-compatible `random_forest_cv.json` (matching `src.random_forest.make_random_forest(**selected_params)`), while keeping `results/` untouched unless explicitly updated.
>>>>>>> Stashed changes
- **Final held-out evaluation.** Once the group approves a frozen protocol and exact run, place `final_protocol.json` in `m3_workspace/` with keys `approved` (true), `run_id`, `dataset_fingerprint`, and `protocol` (the approved evaluation protocol). The named run must be completed and official. The GUI requires confirmation and records a one-time attempt for the dataset fingerprint. Existing final results can be viewed again. A failed attempt stays locked for group review.
- **M2 dependency.** Official runs are blocked if the shared preprocessing contract or official data is unavailable. Sandbox experiments remain available.
