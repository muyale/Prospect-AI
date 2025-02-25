# final_summary.py

import pandas as pd

def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        return None

def read_text_file(file_path):
    """Reads a text file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file '{file_path}': {e}")
        return None

def print_final_summary():
    # Define file paths
    synthetic_data_path = "data/synthetic/lead_data.csv"
    processed_data_path = "data/processed/processed_data.csv"
    scored_data_path = "data/processed/scored_data.csv"
    leads_report_path = "data/final/leads_report.txt"
    performance_report_path = "data/final/performance_report.txt"
    decision_summary_path = "data/final/decision_summary.csv"

    print("\n--- Final Summary ---\n")
    
    # Synthetic Data
    print("Synthetic Data (first 5 rows):")
    df_synthetic = read_csv_file(synthetic_data_path)
    if df_synthetic is not None:
        print(df_synthetic.head())
    else:
        print("File not found:", synthetic_data_path)
    
    print("\nProcessed Data (first 5 rows):")
    df_processed = read_csv_file(processed_data_path)
    if df_processed is not None:
        print(df_processed.head())
    else:
        print("File not found:", processed_data_path)
    
    print("\nScored Data (first 5 rows):")
    df_scored = read_csv_file(scored_data_path)
    if df_scored is not None:
        print(df_scored.head())
    else:
        print("File not found:", scored_data_path)
    
    print("\nLeads Report:")
    leads_report = read_text_file(leads_report_path)
    if leads_report is not None:
        print(leads_report)
    else:
        print("File not found:", leads_report_path)
    
    print("\nPerformance Report:")
    performance_report = read_text_file(performance_report_path)
    if performance_report is not None:
        print(performance_report)
    else:
        print("File not found:", performance_report_path)
    
    print("\nDecision Summary:")
    df_decision = read_csv_file(decision_summary_path)
    if df_decision is not None:
        print(df_decision)
    else:
        print("File not found:", decision_summary_path)

if __name__ == "__main__":
    print_final_summary()
