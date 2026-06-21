import pandas as pd
import numpy as np
import os
import json

def generate_quality_report(report_data):
    report_path = "data/processed/data_quality_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Data Quality Report\n\n")
        
        f.write("## Synthea Encounters Cleaning\n")
        f.write(f"- Initial encounters: {report_data['initial_encounters']}\n")
        f.write(f"- Acute stays (inpatient/urgentcare/emergency): {report_data['acute_stays']}\n")
        f.write(f"- Encounters with missing discharge times imputed: {report_data['imputed_discharges']}\n")
        f.write(f"- 99th percentile ALOS threshold (days): {report_data['alos_threshold']:.1f}\n")
        f.write(f"- Outliers dropped (ALOS > 99th pct): {report_data['outliers_dropped']}\n")
        f.write(f"- Final cleaned encounters: {report_data['final_encounters']}\n\n")
        
        f.write("## CMS & HCRIS Cleaning\n")
        f.write(f"- Total Hospitals Processed (CMS): {report_data['cms_hospitals']}\n")
        f.write(f"- Total Hospitals Processed (HCRIS): {report_data['hcris_hospitals']}\n")
    
    print(f"\n--- Data Quality Report generated at {report_path} ---")

def clean_synthea_data():
    print("Cleaning Synthea Encounters Data...")
    encounters_path = "data/raw/synthea/csv/encounters.csv"
    
    # If the file doesn't exist yet (still generating), we use a fallback logic or mock it for script testing
    if not os.path.exists(encounters_path):
        print(f"Warning: {encounters_path} not found. The generation script may still be running.")
        print("Falling back to our mocked 'data/patients.csv' for cleaning demonstration.")
        
        # Fallback using our Phase 1 mock data just to demonstrate the logic requested
        df = pd.read_csv("data/patients.csv")
        # Rename columns to mimic our needed structure for this demo
        df.rename(columns={'length_of_stay': 'ALOS', 'admission_date': 'START', 'discharge_date': 'STOP'}, inplace=True)
        # Ensure START and STOP are datetime
        df['START'] = pd.to_datetime(df['START'])
        df['STOP'] = pd.to_datetime(df['STOP'])
        df['ENCOUNTERCLASS'] = 'inpatient' # Fake it to all acute
        df['REASONDESCRIPTION'] = df['condition']
        df['PROVIDER'] = "100000" # Dummy provider
    else:
        df = pd.read_csv(encounters_path, low_memory=False)
    
    report = {}
    report['initial_encounters'] = len(df)
    
    # 1. Filter to acute stays (inpatient)
    valid_classes = ['inpatient', 'emergency', 'urgentcare']
    if 'ENCOUNTERCLASS' in df.columns:
        df = df[df['ENCOUNTERCLASS'].isin(valid_classes)].copy()
    report['acute_stays'] = len(df)
    
    # 2. Parse dates and compute ALOS
    if 'START' in df.columns and 'STOP' in df.columns:
        df['START'] = pd.to_datetime(df['START'], errors='coerce')
        df['STOP'] = pd.to_datetime(df['STOP'], errors='coerce')
        
        # 3. Impute missing discharge times using department-level medians
        # We'll use ENCOUNTERCLASS as a proxy for department if department isn't available
        group_col = 'department' if 'department' in df.columns else 'ENCOUNTERCLASS'
        
        # Count missing before
        missing_stop = df['STOP'].isna().sum()
        report['imputed_discharges'] = int(missing_stop)
        
        if missing_stop > 0:
            # Calculate ALOS for rows WITH a stop date to find medians
            temp_alos = (df['STOP'] - df['START']).dt.days
            medians = temp_alos.groupby(df[group_col]).median()
            
            # For rows with NaT in STOP, add the median days to START
            for dept, median_val in medians.items():
                if pd.isna(median_val): median_val = 1
                mask = df['STOP'].isna() & (df[group_col] == dept)
                df.loc[mask, 'STOP'] = df.loc[mask, 'START'] + pd.to_timedelta(median_val, unit='d')
        
        # Compute final ALOS
        df['ALOS'] = (df['STOP'] - df['START']).dt.total_seconds() / (24 * 3600)
        # Handle cases where ALOS might be 0 or negative (same day discharge -> ALOS = 0.5)
        df.loc[df['ALOS'] <= 0, 'ALOS'] = 0.5
    
    # 4. Outlier detection (> 99th percentile)
    if 'ALOS' in df.columns:
        threshold = df['ALOS'].quantile(0.99)
        report['alos_threshold'] = threshold
        
        outliers = df['ALOS'] > threshold
        report['outliers_dropped'] = int(outliers.sum())
        
        df = df[~outliers]
        
    report['final_encounters'] = len(df)
    
    # Write to parquet
    os.makedirs('data/processed', exist_ok=True)
    df.to_parquet('data/processed/encounters_cleaned.parquet', index=False)
    
    return report

def clean_cms_hcris_data(report):
    print("Cleaning CMS & HCRIS Data...")
    
    # CMS Hospital Data
    cms_path = "data/raw/cms_hospital_info.csv"
    if os.path.exists(cms_path):
        df_cms = pd.read_csv(cms_path, low_memory=False)
        # Standardize CCN (Facility ID) to 6 digits zero-padded
        if 'Facility ID' in df_cms.columns:
            df_cms['CCN'] = df_cms['Facility ID'].astype(str).str.zfill(6)
        report['cms_hospitals'] = len(df_cms)
        df_cms.to_parquet('data/processed/cms_hospital_info_cleaned.parquet', index=False)
    else:
        report['cms_hospitals'] = 0
        
    # HCRIS Staffing Mock Data
    hcris_path = "data/raw/hcris_staffing_summary.csv"
    if os.path.exists(hcris_path):
        df_hcris = pd.read_csv(hcris_path)
        # Standardize CCN
        if 'provider_id' in df_hcris.columns:
            df_hcris['CCN'] = df_hcris['provider_id'].astype(str).str.zfill(6)
            
        # Normalize staffing ratios to a common FTE-per-bed metric
        if 'total_nurses' in df_hcris.columns and 'total_beds' in df_hcris.columns:
            # Add safety check against divide by zero
            df_hcris['FTE_per_bed'] = df_hcris['total_nurses'] / df_hcris['total_beds'].replace(0, 1)
            
        report['hcris_hospitals'] = len(df_hcris)
        df_hcris.to_parquet('data/processed/hcris_staffing_cleaned.parquet', index=False)
    else:
        report['hcris_hospitals'] = 0
        
    return report

def main():
    report = clean_synthea_data()
    report = clean_cms_hcris_data(report)
    generate_quality_report(report)

if __name__ == '__main__':
    main()
