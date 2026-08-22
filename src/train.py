import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from src.data_loader import load_data, split_features_target
from src.preprocessor import create_preprocessor

def train_and_evaluate(data_path: str = 'verizonet_data.csv', model_output_path: str = 'models/churn_prediction_pipeline.joblib'):
    """
    Executes end-to-end model training, cross-validation benchmarking, test evaluation,
    and serializes the top-performing pipeline artifact.
    """
    print(f"[*] Loading dataset from '{data_path}'...")
    raw_df = load_data(data_path)
    X, y, num_cols, cat_cols = split_features_target(raw_df)
    
    print(f"[*] Dataset shape: {raw_df.shape[0]} rows, {len(num_cols)} numerical features, {len(cat_cols)} categorical features.")
    print(f"[*] Class distribution: {np.bincount(y)} (Churn rate: {y.mean()*100:.2f}%)")
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    candidate_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_pipelines = {}
    
    print("\n" + "="*70)
    print("                MODEL BENCHMARKING (5-Fold Stratified CV)")
    print("="*70)
    
    for name, model in candidate_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        cv_res = cross_validate(pipe, X_train, y_train, cv=skf, scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
        
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
        
        results.append({
            'Model': name,
            'CV Acc': f"{cv_res['test_accuracy'].mean()*100:.2f}%",
            'CV F1': f"{cv_res['test_f1'].mean():.3f}",
            'CV ROC-AUC': f"{cv_res['test_roc_auc'].mean():.3f}",
            'Test Acc': f"{accuracy_score(y_test, y_pred)*100:.2f}%",
            'Test Precision': f"{precision_score(y_test, y_pred):.3f}",
            'Test Recall': f"{recall_score(y_test, y_pred):.3f}",
            'Test F1': f"{f1_score(y_test, y_pred):.3f}",
            'Test ROC-AUC': f"{roc_auc_score(y_test, y_prob):.3f}"
        })
        
    benchmark_df = pd.DataFrame(results)
    print(benchmark_df.to_string(index=False))
    print("="*70)
    
    # Champion model: Gradient Boosting
    champion_name = 'Gradient Boosting'
    champion_pipeline = trained_pipelines[champion_name]
    
    print(f"\n[+] Champion Model Selected: {champion_name}")
    y_test_pred = champion_pipeline.predict(X_test)
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_test_pred, target_names=['Retained (0)', 'Churned (1)']))
    
    print("--- Test Set Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
    
    # Export model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(champion_pipeline, model_output_path)
    print(f"\n[+] Production model successfully serialized to: '{model_output_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Customer Churn Prediction Model")
    parser.add_argument('--data', type=str, default='verizonet_data.csv', help='Path to CSV dataset')
    parser.add_argument('--output', type=str, default='models/churn_prediction_pipeline.joblib', help='Path to output model artifact')
    args = parser.parse_args()
    
    train_and_evaluate(args.data, args.output)
