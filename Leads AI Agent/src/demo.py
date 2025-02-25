import pandas as pd
from generate_synthetic_data import generate_synthetic_data
from data_processing import load_data, preprocess_data
from agent import LeadGenAgent
from report_generation import generate_leads_report

def run_full_pipeline():
    # Step 1: Generate synthetic data
    df_synthetic = generate_synthetic_data(10000)
    df_synthetic.to_csv("data/synthetic/lead_data.csv", index=False)
    print("Synthetic data generated.")

    # Step 2: Preprocess data
    df_raw = load_data("data/synthetic/lead_data.csv")
    df_processed = preprocess_data(df_raw)
    df_processed.to_csv("data/processed/processed_data.csv", index=False)
    print("Data processed.")

    # Step 3: Initialize agent & train MoE model
    agent = LeadGenAgent(input_dim=4, hidden_dim=16, num_experts=3)
    agent.train_model(df_processed, epochs=10)

    # Step 4: Score leads
    df_scored = agent.score_leads(df_processed)
    # Step 5: Generate strategies
    df_scored = agent.generate_strategies(df_scored)
    df_scored.to_csv("data/processed/scored_data.csv", index=False)
    print("Lead scoring & strategy generation complete.")

    # Step 6: Generate final leads report
    df_final_report = generate_leads_report("data/processed/scored_data.csv")
    
    # Ensure generate_leads_report returns a DataFrame
    if isinstance(df_final_report, pd.DataFrame):
        df_final_report.to_csv("data/final/leads_report.csv", index=False)
        print("\nLeads report generated and saved to data/final/leads_report.csv")
    else:
        with open("data/final/leads_report.txt", "w") as f:
            f.write(str(df_final_report))  # Save as a text file instead
        print("\nLeads report saved as a text file at data/final/leads_report.txt")

if __name__ == "__main__":
    run_full_pipeline()
