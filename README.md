# Loan Approval Analysis (ML_Project)

**Business Problem Summary**
- Goal: predict `LoanApproved` decisions using applicant and application features.
- Context: focus on pre-decision variables; exclude post-decision features to avoid target leakage.

**Dataset Overview**
- Source: data file at [data/Loan.csv](data/Loan.csv).
- Contents: borrower demographics, credit metrics, income and debt ratios, and application metadata.
- Target: `LoanApproved` (classification).
- Notable fields and handling:
	- `ApplicationDate`: dropped (artificial/future dates; not relevant to approval logic).
	- `InterestRate`, `BaseInterestRate`, `MonthlyLoanPayment`: dropped (post-decision values not suitable for modeling approval).
	- `RiskScore`: removed (out of scope for the approval problem).
	- `TotalDebtToIncomeRatio`: validated; out-of-range values flagged and filtered.

**Approach & Methodology**
- EDA: inspected schema (`df.info()`), duplicates, unique counts, correlations, and distributions (seaborn/matplotlib).
- Date handling: parsed `ApplicationDate`, observed non-realistic/future dates, then excluded the column.
- Consistency checks: cross-checked `Experience` against `Age`; reviewed DTI validity.
- Feature selection: kept pre-decision features; removed post-decision and out-of-scope columns.
- Feature types: separated `numerical_cols` and `categorical_cols` for downstream modeling.

**Key Insights**
- Post-decision features (interest, payment) introduce leakage; they were removed.
- `ApplicationDate` contained artificial/future dates; excluded from analysis.
- `TotalDebtToIncomeRatio` outside realistic bounds identified; invalid rows filtered.
- `RiskScore` excluded to keep scope aligned with approval modeling.
- Distributions and correlations reviewed to understand feature relationships.

**Final Model Performance**
- Models evaluated: KNN (with `StandardScaler` pipeline), Decision Tree, Random Forest.
- Cross-validation (5-fold, `StratifiedKFold`) ROC-AUC:
	- KNN: 0.891
	- Decision Tree: 0.925
	- Random Forest: 0.954
- Holdout test metrics (best shown by Random Forest):
	- Random Forest — ROC-AUC: 0.958, Accuracy: 0.903, Precision: 0.842, Recall: 0.730, F1: 0.782
	- Decision Tree — ROC-AUC: 0.933, Accuracy: 0.879, Precision: 0.771, Recall: 0.706, F1: 0.737
	- KNN — ROC-AUC: 0.893, Accuracy: 0.870, Precision: 0.806, Recall: 0.598, F1: 0.687
- Final choice: Random Forest (balanced performance and highest ROC-AUC). Confusion matrices and classification reports per model are plotted in the notebook.

**How to Run the Project**
- Prerequisites: Python 3.9+ and pip.
- Install dependencies:
	- `python -m venv .venv && source .venv/bin/activate`
	- `pip install pandas numpy seaborn matplotlib scikit-learn jupyter`
- Open and run the notebook:
	- In VS Code: open [notebook/ML_Project_HannanButt.ipynb](notebook/ML_Project_HannanButt.ipynb) and run cells.
	- Or via Jupyter: `jupyter lab` and navigate to the notebook.
- Ensure the notebook loads data from the local path [data/Loan.csv](data/Loan.csv) (update any Colab paths if present).

**Project Structure**
- Root files:
	- [README.md](README.md)
	- [data/Loan.csv](data/Loan.csv)
	- [notebook/ML_Project_HannanButt.ipynb](notebook/ML_Project_HannanButt.ipynb)

**Notes**
- Libraries used in the notebook: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`.
- Feature engineering highlights: ordinal mapping for `EducationLevel`, one-hot encoding for selected categoricals (`EmploymentStatus`, `MaritalStatus`, `HomeOwnershipStatus`, `LoanPurpose`), and `train_test_split` with stratification.
- If you later add models, consider saving trained artifacts (e.g., `models/`) and adding a `requirements.txt` for reproducibility.