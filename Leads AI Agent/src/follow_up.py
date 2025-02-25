import pandas as pd
from transformers import pipeline
import os
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

def load_llm_for_follow_up():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = getpass("Enter your Hugging Face token (press Enter if not needed): ")
    llm = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b",
        use_auth_token=hf_token if hf_token else None
    )
    return llm

def generate_follow_up_templates(df: pd.DataFrame):
    llm = load_llm_for_follow_up()
    templates = []
    for _, row in df.iterrows():
        prompt = (
            f"Generate a follow-up email template for a lead in the {row['industry']} industry, "
            f"with a lead score of {row['lead_score']:.2f}, who has visited the website {row['website_visits']} times "
            f"and clicked {row['email_clicks']} emails. Include personalized recommendations and a strong call-to-action."
        )
        result = llm(prompt, max_length=150, num_return_sequences=1)
        templates.append(result[0]['generated_text'])
    df['follow_up_template'] = templates
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/scored_data.csv")
    df = generate_follow_up_templates(df)
    df.to_csv("data/final/follow_up_templates.csv", index=False)
    print("Follow-up templates generated and saved to data/final/follow_up_templates.csv")
