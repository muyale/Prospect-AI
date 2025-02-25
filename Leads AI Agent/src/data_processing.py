import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace missing values
    df = df.fillna(0)
    
    # Compute derived metrics
    df['ad_click_through_rate'] = df['ad_clicks'] / df['ad_impressions']
    df['conversion_rate'] = df['conversions'] / df['ad_clicks']
    
    # Replace infinities and fill NAs explicitly (avoiding inplace chaining)
    df['ad_click_through_rate'] = df['ad_click_through_rate'].replace([float('inf')], 0)
    df['ad_click_through_rate'] = df['ad_click_through_rate'].fillna(0)
    
    df['conversion_rate'] = df['conversion_rate'].replace([float('inf')], 0)
    df['conversion_rate'] = df['conversion_rate'].fillna(0)
    
    return df

if __name__ == "__main__":
    df = load_data("data/synthetic/lead_data.csv")
    df = preprocess_data(df)
    df.to_csv("data/processed/processed_data.csv", index=False)
    print("Data processed and saved to data/processed/processed_data.csv")
