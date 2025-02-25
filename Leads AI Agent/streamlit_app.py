import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import time
import os

from src.performance_analytics import (
    generate_performance_report,
    plot_lead_score_distribution,
    plot_website_visits_vs_lead_score,
    plot_correlation_heatmap
)
from src.decision import cluster_leads, generate_cluster_summary, generate_decision_summary

# Set page configuration
st.set_page_config(page_title="Prospect Pro AI - AI For Lead Generation")

# --- Title & Image ---
st.title("Prospect Pro AI - Dashboard")
try:
    image = Image.open("images/lead-typology-b2b-b2c-cold-hot.webp")
    st.image(image, caption="Overview of the Lead Generation Process", use_container_width=True)
except Exception:
    st.warning("Dashboard image not found in the images folder.")

# --- Introduction Section ---
st.markdown("""
## About Prospect Pro AI

Prospect Pro AI is an advanced, data-driven system designed to revolutionize lead generation and marketing. Powered by a state-of-the-art Mixture-of-Experts (MoE) neural network, it handles complex challenges by:
- Generating synthetic business and advertising data to simulate real-world customer interactions.
- Processing and scoring leads using sophisticated machine learning techniques.
- Applying advanced analytics and clustering to segment leads and extract actionable insights.
- Providing comprehensive performance reports and strategic recommendations to optimize marketing efforts and boost sales.

This system delivers real-time visualizations, detailed reports, and decision support to empower data-driven decision making.
""")

# --- Run Demo Pipeline Button ---
if st.button("Run Prospect Pro AI"):
    with st.spinner("Running the demo pipeline. Please wait..."):
        result = subprocess.run(["python", "src/demo.py"], capture_output=True, text=True)
        st.text(result.stdout)
        if result.returncode == 0:
            st.success("Demo pipeline completed successfully.")
        else:
            st.error("Pipeline encountered an error. Please check the logs.")
    time.sleep(2)  # Allow time for file updates

# --- Load and Display Leads Report (Text) ---
st.markdown("## Leads Report")
def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

leads_report_path = "data/final/leads_report.txt"
report_text = load_text_file(leads_report_path)
if report_text:
    st.text_area("Generated Leads Report", report_text, height=300)
else:
    st.warning("Leads report not found. Please run the pipeline.")

# --- Load Scored Leads Data ---
scored_data_path = "data/processed/scored_data.csv"
try:
    df_scored = pd.read_csv(scored_data_path)
except FileNotFoundError:
    st.error("No scored leads data found. Please run the pipeline first.")
    df_scored = None

# --- Display Lead Generation Results ---
if df_scored is not None:
    st.markdown("## Lead Generation Results")
    st.markdown("### Top 10 Leads by Score")
    top_10 = df_scored.sort_values('lead_score', ascending=False).head(10)
    st.dataframe(top_10[['company_name', 'industry', 'lead_score', 'strategy']])

# --- Display Performance Analytics ---
st.markdown("## Performance Analytics")
performance_report_path = "data/final/performance_report.txt"
performance_report = load_text_file(performance_report_path)
if performance_report:
    st.markdown("### Performance Report")
    st.text(performance_report)
else:
    st.info("Performance report not found. Please run the pipeline.")

if df_scored is not None:
    st.markdown("### Lead Score Distribution")
    fig_hist = plot_lead_score_distribution(df_scored)
    st.pyplot(fig_hist)
    
    st.markdown("### Website Visits vs. Lead Score")
    fig_scatter = plot_website_visits_vs_lead_score(df_scored)
    st.pyplot(fig_scatter)
    
    st.markdown("### Correlation Heatmap")
    fig_heat = plot_correlation_heatmap(df_scored)
    st.pyplot(fig_heat)

# --- Display Real-Time Decision Summary from decision.py ---
st.markdown("## Decision Summary")
# Generate the decision summary in real time using the decision module:
if df_scored is not None:
    try:
        # Use the scored data to cluster leads and generate a decision summary DataFrame
        df_clustered, _ = cluster_leads(data_path=scored_data_path, n_clusters=3)
        cluster_summary = generate_cluster_summary(df_clustered)
        decision_df = generate_decision_summary(df_clustered, cluster_summary)
        st.dataframe(decision_df)
    except Exception as e:
        st.error(f"Error generating decision summary: {e}")
else:
    st.info("No scored leads data available to generate decision summary.")

# --- Display Advanced Analytics Report ---
st.markdown("## Advanced Analytics Report")
advanced_overall_path = "data/final/overall_summary.csv"
advanced_industry_path = "data/final/industry_summary.csv"

try:
    overall_summary_df = pd.read_csv(advanced_overall_path)
    st.subheader("Overall Summary")
    st.dataframe(overall_summary_df)
except Exception as e:
    st.error(f"Error loading overall summary: {e}")

try:
    industry_summary_df = pd.read_csv(advanced_industry_path)
    st.subheader("Industry Performance Summary")
    st.dataframe(industry_summary_df)
except Exception as e:
    st.error(f"Error loading industry summary: {e}")

