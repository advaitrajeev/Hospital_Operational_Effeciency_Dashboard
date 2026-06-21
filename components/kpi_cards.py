import streamlit as st

def render_kpi(label, value, delta=None, delta_color="normal"):
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)

def render_executive_kpis(df):
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate simple mock KPIs based on data
    total_patients = len(df)
    avg_alos = df['length_of_stay'].mean()
    readmission_rate = df['readmitted_30d'].mean() * 100
    
    # Assume 1000 total beds for utilization math
    mock_bed_utilization = 82.5 # Mock value for display
    
    with col1:
        render_kpi("Bed Utilization", f"{mock_bed_utilization}%", "-2.1%", "inverse")
        
    with col2:
        render_kpi("Avg Length of Stay", f"{avg_alos:.1f} days", "+0.3 days", "inverse")
        
    with col3:
        render_kpi("30-Day Readmission", f"{readmission_rate:.1f}%", "-0.5%", "inverse")
        
    with col4:
        render_kpi("Total Discharges (YTD)", f"{total_patients:,}", "+124", "normal")
