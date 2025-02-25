import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_performance_report(df: pd.DataFrame) -> str:
    """
    Generates a detailed performance report summarizing key metrics.
    """
    total_leads = len(df)
    avg_score = df['lead_score'].mean()
    median_score = df['lead_score'].median()
    std_score = df['lead_score'].std()
    
    # Additional metrics if available
    avg_conversion = df['conversion_rate'].mean() if 'conversion_rate' in df.columns else None

    report_lines = [
        f"Total Leads: {total_leads}",
        f"Average Lead Score: {avg_score:.2f}",
        f"Median Lead Score: {median_score:.2f}",
        f"Standard Deviation of Lead Score: {std_score:.2f}"
    ]
    
    if avg_conversion is not None:
        report_lines.append(f"Average Conversion Rate: {avg_conversion:.2f}")

    report = "\n".join(report_lines)
    return report

def plot_lead_score_distribution(df: pd.DataFrame):
    """
    Generates a histogram for lead score distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['lead_score'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Lead Scores")
    ax.set_xlabel("Lead Score")
    ax.set_ylabel("Frequency")
    return fig

def plot_website_visits_vs_lead_score(df: pd.DataFrame):
    """
    Generates a scatter plot for website visits versus lead score.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['website_visits'], df['lead_score'], alpha=0.5, color='green')
    ax.set_title("Website Visits vs. Lead Score")
    ax.set_xlabel("Website Visits")
    ax.set_ylabel("Lead Score")
    return fig

def plot_boxplot_by_industry(df: pd.DataFrame):
    """
    Generates a box plot of lead scores grouped by industry.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='industry', y='lead_score', data=df, ax=ax)
    ax.set_title("Lead Score Distribution by Industry")
    ax.set_xlabel("Industry")
    ax.set_ylabel("Lead Score")
    plt.xticks(rotation=45, ha="right")
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Generates a correlation heatmap for selected numeric columns.
    """
    numeric_cols = ['website_visits', 'email_clicks', 'ad_impressions', 'ad_clicks', 'conversions', 'lead_score']
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

if __name__ == "__main__":
    df = pd.read_csv("data/processed/scored_data.csv")
    report = generate_performance_report(df)
    with open("data/final/performance_report.txt", "w") as f:
        f.write(report)
    print("Performance report generated and saved to data/final/performance_report.txt")
    
    fig_hist = plot_lead_score_distribution(df)
    fig_scatter = plot_website_visits_vs_lead_score(df)
    fig_box = plot_boxplot_by_industry(df)
    fig_heat = plot_correlation_heatmap(df)
    
    # Save plots for verification (optional)
    fig_hist.savefig("data/analytics/lead_score_distribution.png")
    fig_scatter.savefig("data/analytics/website_vs_score.png")
    fig_box.savefig("data/analytics/boxplot_by_industry.png")
    fig_heat.savefig("data/analytics/correlation_heatmap.png")
    
    print("Plots generated and saved in data/analytics folder.")
