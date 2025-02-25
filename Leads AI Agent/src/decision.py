import pandas as pd
from sklearn.cluster import KMeans
import os

def cluster_leads(data_path="data/processed/scored_data.csv", n_clusters=3):
    """
    Load scored leads data and apply k-means clustering on selected features.
    Returns the DataFrame with an additional 'cluster' column and the k-means model.
    """
    df = pd.read_csv(data_path)
    # Select features for clustering; adjust as needed.
    features = ['website_visits', 'email_clicks', 'ad_click_through_rate', 'conversion_rate', 'lead_score']
    X = df[features].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters
    return df, kmeans

def generate_cluster_summary(df):
    """
    Generate summary statistics for each cluster.
    """
    summary = df.groupby('cluster').agg({
        'lead_score': ['mean', 'median', 'std'],
        'website_visits': 'mean',
        'email_clicks': 'mean',
        'conversion_rate': 'mean'
    })
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

def generate_decision_summary(df, cluster_summary):
    """
    Generate a decision summary DataFrame based on the clustering results.
    This includes the cluster summary statistics and recommendations.
    """
    recommendations = []
    for cluster, row in cluster_summary.iterrows():
        avg_score = row['lead_score_mean']
        if avg_score > 0.75:
            rec = "High potential. Recommend aggressive multi-channel marketing and highly personalized outreach."
        elif avg_score > 0.5:
            rec = "Moderate potential. Recommend targeted campaigns with regular follow-ups and refined messaging."
        else:
            rec = "Low potential. Focus on cost-effective nurturing strategies and periodic engagement."
        recommendations.append(rec)
    
    # Add recommendations as a new column in the cluster summary DataFrame
    cluster_summary['Recommendation'] = recommendations
    
    # Reset index to include the cluster labels as a column
    final_decision_summary = cluster_summary.reset_index()
    return final_decision_summary

if __name__ == "__main__":
    df_clustered, _ = cluster_leads()
    cluster_summary = generate_cluster_summary(df_clustered)
    decision_df = generate_decision_summary(df_clustered, cluster_summary)
    print(decision_df)
    os.makedirs("data/final", exist_ok=True)
    decision_df.to_csv("data/final/decision_summary.csv", index=False)
    print("Decision summary saved to data/final/decision_summary.csv")
