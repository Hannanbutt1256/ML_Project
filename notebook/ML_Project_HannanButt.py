# Imported all important libraries
import pandas as pd
import numpy as np
import pickle 
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os

from pathlib import Path

base_path = Path(__file__).resolve().parent

csv_path = base_path.parent / 'data' / 'Loan.csv'

df = pd.read_csv(csv_path)


df['ApplicationDate'].duplicated().sum()
df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
df.groupby(df['ApplicationDate'].dt.year).size()

df = df.drop(columns=['ApplicationDate'])

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

invalid_dti = df[
    (df['TotalDebtToIncomeRatio'] < 0) |
    (df['TotalDebtToIncomeRatio'] > 1)
]

numerical_cols = numerical_cols.drop(
    ['InterestRate', 'BaseInterestRate', 'MonthlyLoanPayment', 'RiskScore']
)

df = df[df['LoanAmount'] > 0]
df = df[df['LoanDuration'] > 0]
df = df[df['MonthlyDebtPayments'] > 0]

num_cols = ['LoanAmount', 'MonthlyDebtPayments', 'AnnualIncome', 'SavingsAccountBalance', 'MonthlyIncome', 'TotalDebtToIncomeRatio','CreditScore', 'CreditCardUtilizationRate' ,'UtilityBillsPaymentHistory','TotalLiabilities','CheckingAccountBalance','TotalAssets','NetWorth']

for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

for col in num_cols:
    df[col] = np.log1p(df[col])

education_order = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
}

df['EducationLevel'] = df['EducationLevel'].map(education_order)


categorical_ohe_cols = [
    'EmploymentStatus',
    'MaritalStatus',
    'HomeOwnershipStatus',
    'LoanPurpose'
]

df = pd.get_dummies(
    df,
    columns=categorical_ohe_cols,
    drop_first=True
)

base_features = [
    'AnnualIncome',
    'MonthlyIncome',
    'NetWorth',
    'Age',
    'Experience',
    'TotalAssets',
    'CreditScore',
    'LengthOfCreditHistory',
    'LoanAmount',
    'TotalDebtToIncomeRatio',
    'EducationLevel'
]

encoded_cat_features = [
    col for col in df.columns
    if col.startswith((
        'EmploymentStatus_',
        'MaritalStatus_',
        'HomeOwnershipStatus_',
        'LoanPurpose_'
    ))
]

selected_features = base_features + encoded_cat_features

X = df[selected_features]
y = df['LoanApproved']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf
}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring='roc_auc'
    )
    print(name, scores.mean())



results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results[name] = {
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }
model_randomForest = rf.fit(X_train,y_train)

rows = []

for name, res in results.items():
    rows.append({
        "Model": name,
        "ROC-AUC": round(res["ROC_AUC"], 3),
        "Accuracy": round(res["Classification Report"]["accuracy"], 3),
        "Precision": round(res["Classification Report"]["1"]["precision"], 3),
        "Recall": round(res["Classification Report"]["1"]["recall"], 3),
        "F1-score": round(res["Classification Report"]["1"]["f1-score"], 3)
    })

comparison_df = pd.DataFrame(rows)
comparison_df


for name, res in results.items():
    print(f"\n{name} Classification Report")
    print(
        pd.DataFrame(res["Classification Report"]).transpose().round(3)
    )



with open (base_path.parent / "model.pkl","wb") as file:
     pickle.dump(model_randomForest,file)