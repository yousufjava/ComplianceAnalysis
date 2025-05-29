import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import numpy as np
import kaggle
import os

kaggle.api.authenticate()
dataset_name = "atharvasoundankar/big-4-financial-risk-insights-2020-2025"
kaggle.api.dataset_download_files(dataset_name, path=".", unzip=True)

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    df = pd.read_csv(csv_files[0])
else:
    df = pd.read_csv("big4_financial_risk_compliance.csv")

blue_theme = ['#0033A0', '#005EB8', '#B7C9E2', '#6C7A89', '#222222']

st.set_page_config(page_title="Big 4 Risk & Compliance Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Big 4 Financial Risk & Compliance Dashboard")

# Sidebar filters
firm_name = st.sidebar.multiselect("Firm Name", options=df["Firm_Name"].unique(), default=df["Firm_Name"].unique())
year = st.sidebar.multiselect("Year", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
industry_affected = st.sidebar.multiselect("Industry Affected", options=df["Industry_Affected"].unique(), default=df["Industry_Affected"].unique())
ai_used_for_auditing = st.sidebar.multiselect("AI Used for Auditing", options=df["AI_Used_for_Auditing"].unique(), default=df["AI_Used_for_Auditing"].unique())

filtered_df = df[
    df["Firm_Name"].isin(firm_name) &
    df["Year"].isin(year) &
    df["Industry_Affected"].isin(industry_affected) &
    df["AI_Used_for_Auditing"].isin(ai_used_for_auditing)
]

# Key Metrics
st.markdown("### Key Metrics")
col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
col_k1.metric("Total Audits", int(filtered_df["Total_Audit_Engagements"].sum()))
col_k2.metric("Compliance Violations", int(filtered_df["Compliance_Violations"].sum()))
col_k3.metric("Fraud Cases", int(filtered_df["Fraud_Cases_Detected"].sum()))
col_k4.metric("Avg. Employee Workload", f"{filtered_df['Employee_Workload'].mean():.1f}")
avg_revenue_impact = filtered_df.groupby('Year')["Total_Revenue_Impact"].sum().mean()
col_k5.metric("Revenue Loss Per Year (in million)", f"${avg_revenue_impact:,.2f}")

# Compliance & Risk Trends
st.markdown("---")
st.markdown("#### Compliance & Risk Trends")
col1, col2 = st.columns(2)
with col1:
    year_trend = filtered_df.groupby('Year')[['High_Risk_Cases', 'Compliance_Violations']].sum().reset_index()
    year_trend_melted = year_trend.melt(id_vars='Year', value_vars=['High_Risk_Cases', 'Compliance_Violations'],
                                        var_name='Metric', value_name='Count')
    fig_compliance = px.bar(
        year_trend_melted,
        x='Count',
        y='Year',
        color='Metric',
        orientation='h',
        barmode='group',
        color_discrete_sequence=blue_theme[:2],
        title='Compliance & High Risk Cases (Yearly, Horizontal)'
    )
    st.plotly_chart(fig_compliance, use_container_width=True)
with col2:
    firm_compliance = filtered_df.groupby('Firm_Name')[['Compliance_Violations']].sum().reset_index()
    fig_firm_compliance = px.bar(firm_compliance, x='Firm_Name', y='Compliance_Violations', color='Firm_Name',
                                color_discrete_sequence=blue_theme, title='Compliance Violations by Firm')
    st.plotly_chart(fig_firm_compliance, use_container_width=True)

# Fraud & Revenue Impact
st.markdown("---")
st.markdown("#### Fraud & Revenue Impact")
col3, col4 = st.columns(2)
with col3:
    fraud_by_year = filtered_df.groupby('Year')[['Fraud_Cases_Detected']].sum().reset_index()
    fig_fraud_year = px.line(fraud_by_year, x='Year', y='Fraud_Cases_Detected', markers=True,
                            color_discrete_sequence=blue_theme, title='Fraud Cases Detected (Yearly)')
    st.plotly_chart(fig_fraud_year, use_container_width=True)
with col4:
    revenue_loss = filtered_df.groupby('Year')[['Total_Revenue_Impact']].sum().reset_index()
    fig_revenue_loss = px.area(revenue_loss, x='Year', y='Total_Revenue_Impact', color_discrete_sequence=[blue_theme[0]],
                              title='Revenue Loss (Yearly)')
    st.plotly_chart(fig_revenue_loss, use_container_width=True)

# AI & Audit Effectiveness
st.markdown("---")
st.markdown("#### AI Usage & Audit Effectiveness")
col5, col6 = st.columns(2)
with col5:
    fig_audit_effectiveness = px.violin(
        filtered_df,
        x='AI_Used_for_Auditing',
        y='Audit_Effectiveness_Score',
        color='AI_Used_for_Auditing',
        box=True,
        points='all',
        color_discrete_sequence=blue_theme,
        title='Audit Effectiveness Score Distribution by AI Usage'
    )
    st.plotly_chart(fig_audit_effectiveness, use_container_width=True)
with col6:
    ai_year_yes = filtered_df[filtered_df['AI_Used_for_Auditing'] == 'Yes'].groupby('Year').size().reset_index(name='AI_Used_Audit_Count')
    fig_ai_year = px.bar(
        ai_year_yes,
        x='Year',
        y='AI_Used_Audit_Count',
        color_discrete_sequence=[blue_theme[0]],
        title='Audits Using AI (Yearly)'
    )
    st.plotly_chart(fig_ai_year, use_container_width=True)

# Employee Workload Analysis
st.markdown("---")
st.markdown("#### Employee Workload Analysis")
col7, col8 = st.columns(2)
with col7:
    workload_firm = filtered_df.groupby('Firm_Name')[['Employee_Workload']].mean().reset_index()
    fig_workload_firm = px.bar(
        workload_firm,
        x='Employee_Workload',
        y='Firm_Name',
        color='Firm_Name',
        orientation='h',
        color_discrete_sequence=blue_theme,
        title='Avg. Employee Workload by Firm (Horizontal)'
    )
    st.plotly_chart(fig_workload_firm, use_container_width=True)
with col8:
    workload_firm_ai_2025 = filtered_df[filtered_df['Year'] == 2025].groupby(['Firm_Name', 'AI_Used_for_Auditing'])[['Employee_Workload']].mean().reset_index()
    fig_workload_firm_ai_2025 = px.bar(
        workload_firm_ai_2025,
        x='Firm_Name',
        y='Employee_Workload',
        color='AI_Used_for_Auditing',
        barmode='group',
        color_discrete_sequence=blue_theme,
        title='Employee Workload by Firm & AI Usage (2025)'
    )
    st.plotly_chart(fig_workload_firm_ai_2025, use_container_width=True)
