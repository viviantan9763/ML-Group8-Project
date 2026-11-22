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

## Project Structure and File Descriptions

The project repository is organised into the following folders and files to facilitate code review, reproducibility, and evaluation.

| Folder / File | Description |
|---|---|
| `data/raw/` | Contains the original UCI Credit Card Default dataset used in this project. |
| `data/processed/` | Contains the prepared datasets, including the full dataset without the ID column, training dataset (24,000 records), and testing dataset (6,000 records). |
| `notebooks/` | Contains Jupyter Notebooks for exploratory data analysis, preprocessing, model training, ensemble learning, and final evaluation. |
| `src/` | Contains Python source code for data preprocessing, Logistic Regression, Random Forest, Gradient Boosting, ensemble learning, model evaluation, and testing. |
| `results/` | Contains saved experimental results, cross-validation metrics, hyperparameter tuning results, model comparisons, and final evaluation outputs in CSV and JSON formats. |
| `figures/` | Contains visualisations generated during exploratory data analysis and model evaluation. |
| `artifacts/` | Contains saved trained machine learning models for reuse and evaluation. |
| `m3_workspace/` | Contains supporting Random Forest experiments, testing scripts, and development records. |
| `report/` | Contains the final group project report, including methodology, experiments, results, discussions, and references. |
| `requirements.txt` | Lists the Python libraries and dependencies required to execute the project. |
| `README.md` | Provides the project overview, dataset information, team responsibilities, repository structure, and setup instructions. |

### Key Submission Files

The primary deliverables can be located as follows:

| Deliverable | Location |
|---|---|
| **Python Source Code** | `src/` |
| **Processed Dataset** | `data/processed/` |
| **Final Jupyter Notebook** | `notebooks/Group8_Final_ML_ORDERED.ipynb` |
| **Model Results and Evaluation** | `results/` |
| **Final Project Report** | `report/` |
| **Python Dependencies** | `requirements.txt` |

The final Jupyter Notebook consolidates the main machine learning workflow, including exploratory data analysis, preprocessing, base model development, ensemble learning, and final evaluation.

For reproducibility, users should install the dependencies listed in `requirements.txt` before executing the Python scripts or Jupyter Notebooks.

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
