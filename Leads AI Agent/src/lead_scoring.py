import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['feature_sum'] = (
        df['website_visits'] * 0.2 +
        df['email_clicks'] * 0.3 +
        df['ad_click_through_rate'] * 0.2 +
        df['conversion_rate'] * 0.3
    )
    return df

def train_predictive_model(df: pd.DataFrame):
    df['qualified'] = (df['conversion_rate'] > df['conversion_rate'].median()).astype(int)
    X = df[['website_visits', 'email_clicks', 'ad_click_through_rate', 'conversion_rate']]
    y = df['qualified']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Predictive Model Accuracy:", accuracy_score(y_test, preds))
    joblib.dump(model, "data/processed/predictive_model.pkl")
    return model

def score_leads(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[['website_visits', 'email_clicks', 'ad_click_through_rate', 'conversion_rate']]
    df['lead_score'] = model.predict_proba(X)[:, 1]
    return df

def save_leads_report(df: pd.DataFrame, file_path: str):
    """ Saves the lead report as a CSV for better integration with Streamlit. """
    df.to_csv(file_path, index=False)
    print(f"✅ Leads report saved to {file_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    df = create_features(df)
    model = train_predictive_model(df)
    df = score_leads(df, model)
    df.to_csv("data/processed/scored_data.csv", index=False)
    print("Lead scoring complete and saved to data/processed/scored_data.csv")
