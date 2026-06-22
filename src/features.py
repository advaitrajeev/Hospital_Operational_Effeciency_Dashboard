import pandas as pd
import numpy as np
import os
import argparse

def ensure_base_features(df):
    """Ensure mock features generated in EDA are present for pipeline consistency."""
    if 'Payer' not in df.columns:
        payers = ['Medicare', 'Medicaid', 'Commercial', 'Self-Pay']
        np.random.seed(42)
        df['Payer'] = np.random.choice(payers, size=len(df), p=[0.4, 0.2, 0.35, 0.05])
    
    if 'age' not in df.columns:
        np.random.seed(42)
        df['age'] = np.random.normal(65, 15, len(df)).clip(18, 100)
        
    drg_map = {
        'Heart Failure': 'DRG291',
        'Sepsis': 'DRG871',
        'Stroke': 'DRG064',
        'Pneumonia': 'DRG193',
        'Hip Fracture': 'DRG469',
        'Appendicitis': 'DRG339'
    }
    if 'Condition' not in df.columns:
        if 'REASONDESCRIPTION' in df.columns:
            df['Condition'] = df['REASONDESCRIPTION'].fillna('Other')
        else:
            np.random.seed(42)
            df['Condition'] = np.random.choice(list(drg_map.keys()), size=len(df))
            
    if 'DRG' not in df.columns:
        df['DRG'] = df['Condition'].map(drg_map).fillna('DRG999')
        
    return df

def build_readmission_features(df):
    """
    Engineer features specifically for predicting 30-day readmissions.
    """
    print("Building readmission features...")
    features = df.copy()
    np.random.seed(101)
    
    # 1. prior_admissions_90d
    # Real logic: sort by patient and date, count admissions in 90 days.
    # Since we lack true patient longitudinal IDs in flat mock data, we simulate it
    # based on age and condition severity.
    features['prior_admissions_90d'] = np.where(
        (features['age'] > 75) | (features['DRG'].isin(['DRG291', 'DRG871'])),
        np.random.poisson(1.5, size=len(features)),
        np.random.poisson(0.2, size=len(features))
    )
    
    # 2. charlson_comorbidity_index
    # Higher age -> higher base CCI.
    base_cci = (features['age'] - 40) / 10
    base_cci = np.maximum(0, base_cci)
    noise = np.random.normal(0, 1, size=len(features))
    features['charlson_comorbidity_index'] = np.clip(np.round(base_cci + noise), 0, 15).astype(int)
    
    # 3. discharge_to_SNF
    # Skilled Nursing Facility. Highly likely for older patients with high ALOS.
    prob_snf = np.where(
        (features['age'] > 80) & (features.get('ALOS', 1) > 5), 0.6,
        np.where(features['age'] > 65, 0.2, 0.05)
    )
    features['discharge_to_SNF'] = np.random.binomial(1, prob_snf)
    
    # 4. payer_type_encoded
    # Target encodings or one-hot. We'll one-hot encode.
    payer_dummies = pd.get_dummies(features['Payer'], prefix='payer')
    features = pd.concat([features, payer_dummies], axis=1)
    
    # 5. diagnosis_complexity_score
    # Count of active ICD-10 codes at discharge
    base_complexity = np.where(features['DRG'] == 'DRG871', 6, 
                      np.where(features['DRG'] == 'DRG291', 4, 2))
    features['diagnosis_complexity_score'] = np.clip(np.round(base_complexity + np.random.normal(0, 1, len(features))), 1, 15).astype(int)
    
    # Target variable simulation if not present
    if 'readmitted_30d' not in features.columns:
        # Logistic probability based on features
        logits = -3.0 + 0.5 * features['prior_admissions_90d'] + 0.2 * features['charlson_comorbidity_index'] + 1.0 * features['discharge_to_SNF']
        probs = 1 / (1 + np.exp(-logits))
        features['readmitted_30d'] = np.random.binomial(1, probs)
        
    return features

def build_alos_features(df):
    """
    Engineer features specifically for predicting ALOS.
    """
    print("Building ALOS features...")
    features = df.copy()
    np.random.seed(202)
    
    # 1. admission_source
    # Emergency vs elective. Proxy using ENCOUNTERCLASS
    features['is_emergency'] = (features.get('ENCOUNTERCLASS', '') == 'emergency').astype(int)
    
    # 2. drg_weight
    # CMS DRG relative weight (encodes clinical complexity)
    drg_weight_map = {
        'DRG871': 2.5, # Sepsis
        'DRG291': 1.8, # Heart Failure
        'DRG064': 1.6, # Stroke
        'DRG469': 2.0, # Hip Fracture
        'DRG193': 1.2, # Pneumonia
        'DRG339': 0.8, # Appendicitis
        'DRG999': 1.0  # Other
    }
    features['drg_weight'] = features['DRG'].map(drg_weight_map).fillna(1.0)
    
    # 3. weekend_admission
    if 'START' in features.columns:
        features['weekend_admission'] = features['START'].dt.dayofweek.isin([5, 6]).astype(int)
    else:
        features['weekend_admission'] = np.random.binomial(1, 2/7.0, size=len(features))
        
    # 4. icu_hours
    # High for sepsis/stroke
    prob_icu = np.where(features['DRG'].isin(['DRG871', 'DRG064']), 0.4, 0.05)
    in_icu = np.random.binomial(1, prob_icu)
    features['icu_hours'] = in_icu * np.random.exponential(48, size=len(features))
    features['icu_hours'] = features['icu_hours'].round(1)
    
    # 5. num_procedures
    base_proc = np.where(features['DRG'] == 'DRG469', 2, # Hip fracture
                np.where(features['DRG'] == 'DRG339', 1, 0)) # Appendicitis
    noise = np.random.poisson(0.5, size=len(features))
    features['num_procedures'] = base_proc + noise
    
    return features

def main():
    os.makedirs('data/features', exist_ok=True)
    
    # Load cleaned data
    input_path = 'data/processed/encounters_cleaned.parquet'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Ensure Phase 3 has run.")
        return
        
    print(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Ensure datetime parsing if necessary
    if 'START' in df.columns:
        df['START'] = pd.to_datetime(df['START'])
    
    df = ensure_base_features(df)
    
    # Generate Feature Sets
    readmission_df = build_readmission_features(df)
    alos_df = build_alos_features(df)
    
    # Save to feature store
    readmission_path = 'data/features/readmission_features.parquet'
    alos_path = 'data/features/alos_features.parquet'
    
    readmission_df.to_parquet(readmission_path, index=False)
    print(f"Readmission features saved to {readmission_path} | Shape: {readmission_df.shape}")
    
    alos_df.to_parquet(alos_path, index=False)
    print(f"ALOS features saved to {alos_path} | Shape: {alos_df.shape}")
    print("\nPhase 5 Feature Engineering Complete.")

if __name__ == '__main__':
    main()
