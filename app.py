import streamlit as st
import pandas as pd
import xgboost as xgb

from components.kpi_cards import render_executive_kpis
from components.charts import plot_department_throughput, plot_alos_distribution, plot_readmissions_by_condition

st.set_page_config(page_title="Hospital Operational Intelligence", layout="wide", page_icon="🏥")

@st.cache_data
def load_data():
    return pd.read_csv('data/patients.csv')

@st.cache_resource
def load_models():
    alos_model = xgb.Booster()
    alos_model.load_model('models/saved/alos_model.json')
    
    readmission_model = xgb.Booster()
    readmission_model.load_model('models/saved/readmission_model.json')
    
    return alos_model, readmission_model

def main():
    st.title("🏥 Hospital Operational Intelligence Platform")
    
    try:
        df = load_data()
        alos_model, readmission_model = load_models()
    except Exception as e:
        st.error(f"Failed to load data or models. Have you run the generation and training scripts? Error: {e}")
        return

    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Departmental View", "Patient Risk ML Predictor"])
    
    with tab1:
        st.subheader("Hospital-Wide Efficiency Metrics")
        render_executive_kpis(df)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            plot_alos_distribution(df)
        with col2:
            plot_readmissions_by_condition(df)

    with tab2:
        st.subheader("Departmental Bottlenecks & Throughput")
        plot_department_throughput(df)
        
        st.markdown("### Deep Dive: Average Length of Stay by Department")
        dept_alos = df.groupby('department')['length_of_stay'].mean().sort_values(ascending=False)
        st.dataframe(dept_alos, use_container_width=True)

    with tab3:
        st.subheader("Patient Risk ML Predictor")
        st.markdown("Enter patient details below to predict their Length of Stay and 30-Day Readmission Risk.")
        
        with st.form("patient_form"):
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=65)
            department = st.selectbox("Department", df['department'].unique())
            condition = st.selectbox("Condition", df['condition'].unique())
            
            submitted = st.form_submit_button("Predict Risk Profile")
            
            if submitted:
                # Prepare single record for inference
                # Note: Categories must match training
                input_data = pd.DataFrame({
                    'age': [age],
                    'department': [department],
                    'condition': [condition]
                })
                input_data['department'] = input_data['department'].astype('category')
                input_data['condition'] = input_data['condition'].astype('category')
                
                dmatrix_alos = xgb.DMatrix(input_data, enable_categorical=True)
                pred_alos = alos_model.predict(dmatrix_alos)[0]
                
                # For readmission, we need length_of_stay as a feature
                input_data['length_of_stay'] = [pred_alos]
                dmatrix_readm = xgb.DMatrix(input_data, enable_categorical=True)
                pred_readm_prob = readmission_model.predict(dmatrix_readm)[0]
                
                st.markdown("### Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Length of Stay", f"{pred_alos:.1f} Days")
                with col2:
                    risk_level = "High" if pred_readm_prob > 0.15 else "Low"
                    color = "red" if risk_level == "High" else "green"
                    st.metric("30-Day Readmission Risk", f"{pred_readm_prob*100:.1f}%", delta=risk_level, delta_color="inverse")

if __name__ == '__main__':
    main()
