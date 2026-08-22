import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib

def load_model(model_path: str = 'models/churn_prediction_pipeline.joblib'):
    """Load serialized inference pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run 'python -m src.train' first.")
    return joblib.load(model_path)

def predict_customer(input_data: pd.DataFrame, model_path: str = 'models/churn_prediction_pipeline.joblib') -> pd.DataFrame:
    """
    Predict churn probabilities, class labels, and risk levels for input customer records.
    """
    model = load_model(model_path)
    
    # Generate predictions
    probs = model.predict_proba(input_data)[:, 1]
    preds = (probs >= 0.50).astype(int)
    
    results = input_data.copy()
    results['Churn_Probability'] = np.round(probs, 4)
    results['Predicted_Churn'] = preds
    
    def assign_risk(p):
        if p >= 0.70:
            return 'High Risk (Immediate Action)'
        elif p >= 0.40:
            return 'Moderate Risk (Watch & Nurture)'
        else:
            return 'Low Risk (Healthy)'
            
    results['Risk_Tier'] = [assign_risk(p) for p in probs]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Customer Churn Risk")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV or JSON string")
    parser.add_argument('--model', type=str, default='models/churn_prediction_pipeline.joblib', help="Path to saved pipeline")
    parser.add_argument('--output', type=str, default='predictions.csv', help="Path to output CSV")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        df_input = pd.read_csv(args.input)
    else:
        # Try parsing JSON string
        try:
            records = json.loads(args.input)
            if isinstance(records, dict):
                records = [records]
            df_input = pd.DataFrame(records)
        except Exception as e:
            raise ValueError(f"Could not parse input as valid file path or JSON string: {e}")
            
    res = predict_customer(df_input, args.model)
    res.to_csv(args.output, index=False)
    print(f"[+] Prediction completed for {len(res)} record(s). Output saved to: '{args.output}'")
    print(res[['Customer ID', 'Monthly Charge', 'Tenure in Months', 'Churn_Probability', 'Predicted_Churn', 'Risk_Tier']].head())
