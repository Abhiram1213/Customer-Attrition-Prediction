"""
Quickstart Demonstration Script
Runs a sample evaluation and customer risk scoring using the trained production model.
"""

import sys
import pandas as pd
from src.predict import predict_customer

def run_demo():
    print("=" * 75)
    print("   TELECOM CUSTOMER ATTRITION PREDICTION & RETENTION DEMO")
    print("=" * 75)
    
    # Load 5 sample customers from the dataset
    df = pd.read_csv('verizonet_data.csv').head(5)
    
    print(f"\n[*] Scoring {len(df)} sample customer profiles through production pipeline...\n")
    results = predict_customer(df)
    
    cols_to_show = [
        'Customer ID', 'Contract', 'Total Customer Svc Requests',
        'Customer Satisfaction rate', 'Monthly Charge', 'Churn_Probability', 'Risk_Tier'
    ]
    
    display_df = results[cols_to_show].copy()
    display_df['Churn_Probability'] = (display_df['Churn_Probability'] * 100).map("{:.1f}%".format)
    
    print(display_df.to_string(index=False))
    
    print("\n" + "=" * 75)
    print("   RECOMMENDED RETENTION ACTIONS")
    print("=" * 75)
    for _, row in results.iterrows():
        print(f"\n[Customer {row['Customer ID']}] - Risk Tier: {row['Risk_Tier']} ({row['Churn_Probability']*100:.1f}% Churn Prob)")
        if row['Churn_Probability'] >= 0.70:
            print("  --> Action: Priority Outreach! Senior agent ticket review + 15% rate lock incentive.")
        elif row['Churn_Probability'] >= 0.40:
            print("  --> Action: Nurture Account. Offer free speed boost or streaming add-on.")
        else:
            print("  --> Action: Account Healthy. Eligible for Referral Reward promotion.")
            
    print("\n" + "=" * 75)
    print("   Demo completed successfully! Explore further in 'Customer Attrition Prediction.ipynb'")
    print("=" * 75)

if __name__ == '__main__':
    run_demo()
