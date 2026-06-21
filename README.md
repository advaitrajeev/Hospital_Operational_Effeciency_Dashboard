# Hospital Operational Intelligence Platform 🏥

## Executive Summary
Hospitals operate at immense scale with razor-thin margins. A 1% improvement in bed utilization at a mid-sized hospital can generate millions in annual revenue. 

The **Hospital Operational Intelligence Platform** is an end-to-end ML-powered Streamlit dashboard designed for Chief Operating Officers (COOs), department heads, and hospital board members. This platform ingests synthetic patient data, models it using machine learning (XGBoost), and provides an interactive overview of the key efficiency drivers of a hospital:
- **Bed Utilization**
- **Average Length of Stay (ALOS)**
- **Readmission Risk (30-day)**
- **Departmental Throughput**

With this dashboard, hospital executives can immediately understand where the hospital is underperforming, identify departmental bottlenecks, and predict readmission risks for individual patients in real-time.

---

## Features
- **Executive Summary Tab:** High-level KPIs and distributions comparing current performance against optimal operational thresholds.
- **Departmental View:** Granular insights showing patient volume and length of stay categorized by department, helping identify operational bottlenecks.
- **Patient Risk ML Predictor:** An interactive machine learning tool where clinicians can input basic demographic and clinical data to instantly predict a patient's expected Length of Stay (ALOS) and their risk of 30-day readmission.

---

## How to Install and Run Locally

### Prerequisites
- Python 3.10+
- Git

### 1. Setup the Environment
First, clone the project (if applicable) and navigate to the project directory:
```bash
cd hospital_operational_effeciency_dashboard
```

Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train Models
Since the application uses simulated patient records for privacy (avoiding PHI), you must generate the synthetic dataset and train the ML models before launching the app.

Generate the synthetic patient data (creates `data/patients.csv`):
```bash
python data/generate_synthea.py
```

Train the ML models for ALOS and Readmission prediction:
```bash
python models/train_alos.py
python models/train_readmission.py
```

### 3. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
This will open the dashboard in your default web browser at `http://localhost:8501`.

---

## Technical Architecture
- **Frontend & Data Visualization:** [Streamlit](https://streamlit.io/), [Plotly Express](https://plotly.com/python/plotly-express/)
- **Machine Learning Models:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
- **Data Generation:** [Faker](https://faker.readthedocs.io/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)

## Business Impact
Predicting and preventing readmissions directly improves Medicare Star Ratings and avoids CMS financial penalties under the Hospital Readmissions Reduction Program. Furthermore, optimizing Average Length of Stay increases the discharge rate per occupied bed, directly driving "Revenue per available bed" (RevPAB).
