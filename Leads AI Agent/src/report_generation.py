import pandas as pd

def generate_leads_report(scored_csv: str) -> str:
    df = pd.read_csv(scored_csv)
    top_leads = df.sort_values('lead_score', ascending=False).head(5)

    report = "Final Leads Report:\n\n"
    report += "Top 5 Leads by Score:\n"
    report += top_leads[['company_name', 'industry', 'lead_score', 'strategy']].to_string(index=False)
    report += "\n\nSuggested Next Steps:\n"
    report += (
        "- Allocate more resources to high-scoring leads.\n"
        "- Conduct personalized outreach for mid-range leads.\n"
        "- Use automated nurture campaigns for low-scoring leads.\n"
        "- Continuously refine the MoE model with new data.\n"
    )
    return report

if __name__ == "__main__":
    final_report = generate_leads_report("data/processed/scored_data.csv")
    with open("data/final/leads_report.txt", "w") as f:
        f.write(final_report)
    print("Leads report generated and saved to data/final/leads_report.txt")
    print("\nGenerated Leads Report:\n", final_report)
