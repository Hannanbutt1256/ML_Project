import pandas as pd
import numpy as np
import argparse
import pickle
import wandb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def load_and_preprocess(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Preprocessing - Clipping and Log Transformation
    num_cols = ['LoanAmount', 'MonthlyDebtPayments', 'AnnualIncome', 'SavingsAccountBalance', 'MonthlyIncome', 'TotalDebtToIncomeRatio','CreditScore', 'CreditCardUtilizationRate' ,'UtilityBillsPaymentHistory','TotalLiabilities','CheckingAccountBalance','TotalAssets','NetWorth']
    for col in num_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
        df[col] = np.log1p(df[col])
    
    # Preprocessing - Categorical Mapping
    education_order = {
        'High School': 0,
        'Associate': 1,
        'Bachelor': 2,
        'Master': 3,
        'Doctorate': 4
    }
    if 'EducationLevel' in df.columns:
        df['EducationLevel'] = df['EducationLevel'].map(education_order)
    
    # Preprocessing - One Hot Encoding
    categorical_ohe_cols = ['EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
    df = pd.get_dummies(df, columns=categorical_ohe_cols, drop_first=True)
    
    # Feature Selection
    base_features = [
        'AnnualIncome', 'MonthlyIncome', 'NetWorth', 'Age', 'Experience', 
        'TotalAssets', 'CreditScore', 'LengthOfCreditHistory', 'LoanAmount', 
        'TotalDebtToIncomeRatio', 'EducationLevel'
    ]
    encoded_cat_features = [col for col in df.columns if col.startswith(('EmploymentStatus_', 'MaritalStatus_', 'HomeOwnershipStatus_', 'LoanPurpose_'))]
    selected_features = base_features + encoded_cat_features
    
    # Ensure all selected features exist (in case some categories are missing in small batches)
    # But for a single CSV load, it should be consistent.
    
    X = df[selected_features]
    y = df['LoanApproved']
    
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rf', choices=['knn', 'dt', 'rf'], help='Model type: knn, dt, or rf')
    # KNN Hyperparams
    parser.add_argument('--n_neighbors', type=int, default=5)
    # DT Hyperparams
    parser.add_argument('--max_depth', type=int, default=10)
    # RF Hyperparams
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(project="loan-approval-prediction", config=vars(args))
    config = wandb.config

    try:
        X, y = load_and_preprocess('data/Loan.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try a different path if running from a different directory
        X, y = load_and_preprocess(os.path.join(os.path.dirname(__file__), 'data', 'Loan.csv'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.random_state)

    if config.model_type == 'knn':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=config.n_neighbors))
        ])
    elif config.model_type == 'dt':
        model = DecisionTreeClassifier(max_depth=config.max_depth, random_state=config.random_state)
    else: # rf
        model = RandomForestClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=config.random_state, n_jobs=-1)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # Console output
    print(f"Model: {config.model_type}")
    print(f"Accuracy: {acc}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # W&B Logging
    wandb.log({
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": wandb.Table(dataframe=pd.DataFrame(report).transpose().reset_index())
    })

    # Save Model Artifact
    model_name = "model.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(model, f)
    
    artifact = wandb.Artifact(f"trained_model_{config.model_type}", type="model")
    artifact.add_file(model_name)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    main()
