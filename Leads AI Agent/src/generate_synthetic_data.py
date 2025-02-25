import pandas as pd
from faker import Faker
import random

def generate_synthetic_data(num_records: int = 1000) -> pd.DataFrame:
    fake = Faker()
    records = []
    for _ in range(num_records):
        record = {
            "company_name": fake.company(),
            "industry": fake.bs(),
            "company_size": random.choice([10, 50, 100, 500, 1000]),
            "website_visits": random.randint(100, 10000),
            "email_clicks": random.randint(10, 500),
            "ad_impressions": random.randint(1000, 50000),
            "ad_clicks": random.randint(50, 2000),
            "conversions": random.randint(5, 500)
        }
        records.append(record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_synthetic_data(20000)
    df.to_csv("data/synthetic/lead_data.csv", index=False)
    print("Synthetic lead data generated and saved to data/synthetic/lead_data.csv")
