
import os
import time
import pandas as pd

from final_summary import print_final_summary
from generate_synthetic_data import generate_synthetic_data
from data_processing import load_data, preprocess_data
from agent import LeadGenAgent
from report_generation import generate_leads_report
from performance_analytics import generate_performance_report
from decision import cluster_leads, generate_cluster_summary, generate_decision_summary

def main():
    introduction = (
        "Welcome to Prospect AI!\n\n"
        "I am Prospect AI, an advanced data-driven system designed to revolutionize lead generation for businesses. "
        "I address the challenge of identifying and nurturing high-quality leads by leveraging cutting-edge algorithms and "
        "machine learning techniques. Here’s how I work:\n\n"
        "1. I generate synthetic business and advertising data to simulate real-world customer interactions.\n"
        "2. I preprocess and normalize the data to compute key performance indicators such as website visits, email clicks, "
        "ad engagement, and conversion rates.\n"
        "3. I use a state-of-the-art Mixture-of-Experts (MoE) neural network built with PyTorch to score and generate strategies for each lead.\n"
        "4. I apply advanced analytics and clustering to segment the leads and extract actionable insights.\n"
        "5. I generate comprehensive reports, including a leads report, performance analytics, and a decision summary with recommendations.\n\n"
        "By combining these techniques, I help stakeholders optimize resource allocation, refine targeting, and ultimately boost sales.\n"
    )
    print(introduction)
    time.sleep(2)  # Pause for dramatic effect

    # --- Step 1: Generate Synthetic Data ---
    print("Step 1: Getting data...")
    df_synthetic = generate_synthetic_data(1000)
    os.makedirs("data/synthetic", exist_ok=True)
    synthetic_data_path = "data/synthetic/lead_data.csv"
    df_synthetic.to_csv(synthetic_data_path, index=False)
    print("Data  saved to", synthetic_data_path)

    # --- Step 2: Preprocess Data ---
    print("\nStep 2: Preprocessing data...")
    df_raw = load_data(synthetic_data_path)
    df_processed = preprocess_data(df_raw)
    os.makedirs("data/processed", exist_ok=True)
    processed_data_path = "data/processed/processed_data.csv"
    df_processed.to_csv(processed_data_path, index=False)
    print("Data processed and saved to", processed_data_path)

    # --- Step 3: Train MoE Model & Generate Lead Scores and Strategies ---
    print("\nStep 3: Training the MoE model and generating lead scores...")
    agent = LeadGenAgent(input_dim=4, hidden_dim=16, num_experts=3)
    agent.train_model(df_processed, epochs=10)
    df_scored = agent.score_leads(df_processed)
    df_scored = agent.generate_strategies(df_scored)
    scored_data_path = "data/processed/scored_data.csv"
    df_scored.to_csv(scored_data_path, index=False)
    print("Lead scoring and strategy generation complete. Scored data saved to", scored_data_path)

    # --- Step 4: Generate Final Leads Report ---
    print("\nStep 4: Generating final leads report...")
    leads_report = generate_leads_report(scored_data_path)
    os.makedirs("data/final", exist_ok=True)
    # Check if the report is a DataFrame; if so, save as CSV; otherwise, save as a text file.
    leads_report_path = ""
    if isinstance(leads_report, pd.DataFrame):
        leads_report_path = "data/final/leads_report.csv"
        leads_report.to_csv(leads_report_path, index=False)
    else:
        leads_report_path = "data/final/leads_report.txt"
        with open(leads_report_path, "w", encoding="utf-8") as f:
            f.write(leads_report)
    print("Leads report generated and saved to", leads_report_path)

    # --- Step 5: Generate Performance Report ---
    print("\nStep 5: Generating performance report...")
    performance_report = generate_performance_report(df_scored)
    performance_report_path = "data/final/performance_report.txt"
    with open(performance_report_path, "w", encoding="utf-8") as f:
        f.write(performance_report)
    print("Performance report generated and saved to", performance_report_path)

    # --- Step 6: Generate Decision Summary ---
    print("\nStep 6: Generating decision summary...")
    df_clustered, _ = cluster_leads(data_path=scored_data_path, n_clusters=3)
    cluster_summary = generate_cluster_summary(df_clustered)
    decision_summary_df = generate_decision_summary(df_clustered, cluster_summary)
    decision_summary_path = "data/final/decision_summary.csv"
    decision_summary_df.to_csv(decision_summary_path, index=False)
    print("Decision summary generated and saved to", decision_summary_path)

    # --- Final Output Summary ---
    print("\nPipeline execution complete. All outputs have been saved in the 'data' folder.")
    print("\n--- Final Summary ---")
    print("Synthetic Data:", synthetic_data_path)
    print("Processed Data:", processed_data_path)
    print("Scored Data:", scored_data_path)
    print("Leads Report:", leads_report_path)
    print("Performance Report:", performance_report_path)
    print("Decision Summary:", decision_summary_path)


if __name__ == "__main__":
    main()
    time.sleep(2) 
    print_final_summary()