# Credit Card Default Prediction — Machine Learning Group Project

## Project Overview

This project focuses on predicting credit card payment defaults using machine learning techniques. The objective is to develop, optimise, and compare different classification models to identify customers who may default on their next credit card payment.

The project uses the **UCI Default of Credit Card Clients Dataset (#350)**, containing 30,000 customer records and 23 predictive features related to demographics, credit limits, repayment history, billing amounts, and previous payments.

Three machine learning models are implemented:

- Logistic Regression
- Random Forest
- Gradient Boosting

The models are evaluated and optimised using cross-validation and hyperparameter tuning. The two best-performing models are then selected to construct an ensemble model and compared against the individual models.

## Team Responsibilities

| Member | Responsibilities |
|---|---|
| M1 | Exploratory Data Analysis (EDA) and Ensemble Learning |
| M2 | Shared Data Preprocessing and Logistic Regression |
| M3 | Random Forest and Final Model Evaluation |
| M4 | Gradient Boosting, Model Interpretation and Recommendations |

## Dataset

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Dataset File:** `data/raw/credit_default.csv`
- **Records:** 30,000
- **Predictive Features:** 23
- **Target:** `default_next_month`
- **Target Values:** 1 = Default, 0 = Non-default

## Machine Learning Workflow

1. **Data Audit and EDA** — Investigate data quality, feature distributions, correlations, outliers, and class imbalance.
2. **Data Preprocessing** — Prepare numerical and categorical features using a shared preprocessing pipeline.
3. **Base Model Development** — Train Logistic Regression, Random Forest, and Gradient Boosting models.
4. **Hyperparameter Optimisation** — Optimise each model using cross-validation.
5. **Ensemble Learning** — Select the two best-performing models and combine their predictions.
6. **Final Evaluation** — Compare the three base models and the ensemble using classification metrics.
7. **Model Interpretation** — Analyse model performance, important features, limitations, and practical suitability.

## Shared Technical Configuration

To ensure consistency across all models, the project follows a common experimental setup:

- **Train/Test Split:** 80% / 20%, stratified
- **Random State:** 42
- **Cross-validation:** 3-Fold StratifiedKFold
- **Primary Selection Metric:** Positive-class F1-score
- **Preprocessing:** Shared Scikit-learn Pipeline
- **Final Test Set:** Reserved for final evaluation

All models use the same training and testing partitions. Model selection and hyperparameter optimisation are performed using training data only to reduce the risk of data leakage.

## Project Structure

| Folder / File | Description |
|---|---|
| `data/raw/` | Original dataset |
| `data/processed/` | Cleaned dataset |
| `notebooks/` | Jupyter Notebooks for analysis and experiments |
| `src/` | Python source code for preprocessing, modelling and evaluation |
| `figures/` | EDA and evaluation visualisations |
| `results/` | Data audit, cross-validation results and model performance metrics |
| `report/` | Project report and related documents |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and setup instructions |

## Getting Started

**1. Clone the repository**

Clone this GitHub repository and open the project folder in VS Code.

**2. Set up a Python environment**

**2. Set up a Python environment**

Ensure that Python is installed on your computer. Open a terminal in the project's root directory and install the required Python dependencies using the provided `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

This command installs the libraries required to run the preprocessing pipeline, machine learning models, evaluation scripts, and Jupyter Notebooks.

**3. Verify shared preprocessing**

From the project root, run:

`python -m src.test_preprocessing`

This checks the compatibility of the shared preprocessing pipeline with Logistic Regression, Random Forest, and Gradient Boosting.

**4. Run the analysis and models**

The EDA Notebook is located in `notebooks/`. Model development and evaluation scripts are maintained in `src/`.

Use the project Python environment and run the relevant scripts as they become available.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Git and GitHub

## References

Yeh, I. (2009). *Default of credit card clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
