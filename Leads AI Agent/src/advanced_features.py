import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

def compute_industry_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group leads by industry and compute key performance metrics.
    Returns a DataFrame with:
      - Average Lead Score
      - Median Lead Score
      - Standard Deviation of Lead Score
      - Average Website Visits
      - Average Email Clicks
      - Average Conversion Rate
    """
    performance = df.groupby('industry').agg({
        'lead_score': ['mean', 'median', 'std'],
        'website_visits': 'mean',
        'email_clicks': 'mean',
        'conversion_rate': 'mean'
    })
    performance.columns = ['_'.join(col).strip() for col in performance.columns.values]
    performance = performance.reset_index()
    return performance

def compute_lead_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Lead Efficiency Index (LEI) for each lead.
    LEI is defined as:
         LEI = (conversion_rate * lead_score) / (website_visits + email_clicks + 1)
    The function returns the DataFrame with two new columns: 'lead_efficiency' and its normalized version.
    """
    df = df.copy()
    df['lead_efficiency'] = (df['conversion_rate'] * df['lead_score']) / (df['website_visits'] + df['email_clicks'] + 1)
    # Normalize the efficiency index
    min_val = df['lead_efficiency'].min()
    max_val = df['lead_efficiency'].max()
    df['lead_efficiency_norm'] = (df['lead_efficiency'] - min_val) / (max_val - min_val + 1e-6)
    return df

def generate_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an overall summary DataFrame with high-level metrics.
    """
    total_leads = len(df)
    avg_efficiency = df['lead_efficiency_norm'].mean()
    median_efficiency = df['lead_efficiency_norm'].median()
    
    overall_summary = pd.DataFrame({
        "Metric": ["Total Leads", "Average Lead Efficiency (Normalized)", "Median Lead Efficiency (Normalized)"],
        "Value": [total_leads, round(avg_efficiency, 2), round(median_efficiency, 2)]
    })
    return overall_summary

def generate_industry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates industry performance summary with recommendations.
    """
    industry_summary = compute_industry_performance(df)
    # For each industry, generate a recommendation based on average lead score
    recommendations = []
    for _, row in industry_summary.iterrows():
        avg_score = row['lead_score_mean']
        if avg_score > 0.75:
            rec = "High potential. Recommend aggressive multi-channel marketing and highly personalized outreach."
        elif avg_score > 0.5:
            rec = "Moderate potential. Recommend targeted campaigns with regular follow-ups and refined messaging."
        else:
            rec = "Low potential. Focus on cost-effective nurturing strategies and periodic engagement."
        recommendations.append(rec)
    
    industry_summary["Recommendation"] = recommendations
    return industry_summary

def generate_advanced_report_df(scored_csv: str) -> dict:
    """
    Loads the scored leads data, computes additional metrics, and returns a dictionary of DataFrames:
      - 'overall_summary': High-level overall summary table.
      - 'industry_summary': Industry performance summary with recommendations.
      - 'leads_with_efficiency': The original data with computed lead efficiency metrics.
    """
    df = pd.read_csv(scored_csv)
    df = compute_lead_efficiency(df)
    overall_summary = generate_overall_summary(df)
    industry_summary = generate_industry_summary(df)
    
    return {
        "overall_summary": overall_summary,
        "industry_summary": industry_summary,
        "leads_with_efficiency": df
    }

if __name__ == "__main__":
    try:
        df_scored = pd.read_csv("data/processed/scored_data.csv")
    except FileNotFoundError:
        print("No scored data found. Please run the pipeline first.")
        exit(1)
    
    advanced_results = generate_advanced_report_df("data/processed/scored_data.csv")
    
    # Save the overall summary and industry summary as CSV files
    os.makedirs("data/final", exist_ok=True)
    advanced_results["overall_summary"].to_csv("data/final/overall_summary.csv", index=False)
    advanced_results["industry_summary"].to_csv("data/final/industry_summary.csv", index=False)
    
    # Also, create a combined text report (optional)
    report_text = (
        "Advanced Lead Generation Analysis Report:\n\n"
        "Overall Summary:\n" + advanced_results["overall_summary"].to_string(index=False) + "\n\n"
        "Industry Performance Summary:\n" + advanced_results["industry_summary"].to_string(index=False) + "\n\n"
        "These insights help stakeholders optimize resource allocation, refine targeting, and improve overall lead quality."
    )
    
    with open("data/final/advanced_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("Advanced report generated and saved to data/final/advanced_report.txt")
    print("Overall Summary:\n", advanced_results["overall_summary"])
    print("Industry Summary:\n", advanced_results["industry_summary"])
