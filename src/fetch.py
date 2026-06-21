import os
import json
import urllib.request
import subprocess
import datetime
import pandas as pd

MANIFEST_FILE = 'data/raw/data_manifest.json'

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {"datasets": {}}

def save_manifest(manifest):
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

def download_cms_data(manifest):
    print("Downloading CMS Hospital Compare Data...")
    # This URL is the Hospital General Information API endpoint
    cms_url = "https://data.cms.gov/provider-data/api/1/datastore/query/xugc-bckj/0/download"
    dest_path = "data/raw/cms_hospital_info.csv"
    
    try:
        urllib.request.urlretrieve(cms_url, dest_path)
        df = pd.read_csv(dest_path)
        row_count = len(df)
        print(f"Downloaded CMS Data: {row_count} rows.")
        
        manifest['datasets']['cms_hospital_info'] = {
            "source_url": cms_url,
            "download_date": datetime.datetime.now().isoformat(),
            "row_count": row_count,
            "schema_version": "1.0",
            "file_path": dest_path
        }
    except Exception as e:
        print(f"Failed to download CMS data: {e}")

def run_synthea(manifest):
    print("Setting up Synthea...")
    jar_url = "https://github.com/synthetichealth/synthea/releases/download/v3.2.0/synthea-with-dependencies.jar"
    jar_path = "synthea.jar"
    
    if not os.path.exists(jar_path):
        print(f"Downloading Synthea JAR from {jar_url}...")
        urllib.request.urlretrieve(jar_url, jar_path)
        
    print("Running Synthea for 50,000 patients. This may take a while...")
    # Using 50000 patients as requested: java -jar synthea.jar -p 50000
    # Note: We set output directory to data/raw/synthea
    cmd = ["java", "-jar", jar_path, "-p", "50000"]
    
    # Configure output to go to data/raw/synthea
    with open("synthea.properties", "w") as f:
        f.write("exporter.baseDirectory = data/raw/synthea\n")
        f.write("exporter.csv.export = true\n")
        f.write("exporter.fhir.export = false\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("Synthea generation complete.")
        
        # Check encounters.csv for row count
        encounters_path = "data/raw/synthea/csv/encounters.csv"
        row_count = "Unknown"
        if os.path.exists(encounters_path):
            df = pd.read_csv(encounters_path, low_memory=False)
            row_count = len(df)
            
        manifest['datasets']['synthea_patients'] = {
            "source_url": "synthea generator v3.2.0",
            "download_date": datetime.datetime.now().isoformat(),
            "row_count": row_count, # Number of encounters
            "schema_version": "3.2.0",
            "file_path": "data/raw/synthea/csv/"
        }
    except Exception as e:
        print(f"Synthea execution failed: {e}")

def download_hcris_mock(manifest):
    print("Generating HCRIS Cost Report Extract Mockup...")
    # Real HCRIS is huge; we'll create a realistic summary dataset
    dest_path = "data/raw/hcris_staffing_summary.csv"
    
    data = []
    for i in range(1000):
        data.append({
            "provider_id": f"{100000 + i}",
            "hospital_name": f"Hospital {i}",
            "total_beds": 100 + (i % 400),
            "total_nurses": 50 + (i % 200),
            "nurse_to_patient_ratio": round(1 / (2 + (i % 4)), 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(dest_path, index=False)
    print(f"Created HCRIS Mock Data: {len(df)} rows.")
    
    manifest['datasets']['hcris_staffing'] = {
        "source_url": "mocked (representing HCRIS 2552-10)",
        "download_date": datetime.datetime.now().isoformat(),
        "row_count": len(df),
        "schema_version": "1.0",
        "file_path": dest_path
    }

def main():
    print("Starting Phase 2 Data Collection...")
    manifest = load_manifest()
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Touch .gitkeep for processed
    open('data/processed/.gitkeep', 'a').close()
    
    download_cms_data(manifest)
    download_hcris_mock(manifest)
    run_synthea(manifest)
    
    save_manifest(manifest)
    print("Data collection pipeline finished. Manifest updated.")

if __name__ == '__main__':
    main()
