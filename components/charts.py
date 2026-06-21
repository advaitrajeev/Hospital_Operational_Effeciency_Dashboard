import streamlit as st
import plotly.express as px

def plot_department_throughput(df):
    dept_counts = df['department'].value_counts().reset_index()
    dept_counts.columns = ['Department', 'Patient Volume']
    fig = px.bar(dept_counts, x='Department', y='Patient Volume', title="Patient Volume by Department", color='Department')
    st.plotly_chart(fig, use_container_width=True)

def plot_alos_distribution(df):
    fig = px.histogram(df, x='length_of_stay', nbins=30, title="Distribution of Length of Stay (Days)")
    st.plotly_chart(fig, use_container_width=True)

def plot_readmissions_by_condition(df):
    readmissions = df.groupby('condition')['readmitted_30d'].mean().reset_index()
    readmissions['readmitted_30d'] *= 100
    readmissions.columns = ['Condition', 'Readmission Rate (%)']
    fig = px.bar(readmissions, x='Condition', y='Readmission Rate (%)', title="Readmission Rates by Condition", color='Condition')
    st.plotly_chart(fig, use_container_width=True)
