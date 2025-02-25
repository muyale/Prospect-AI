import torch
import torch.optim as optim
import pandas as pd
from neural_net import MoEModel
from sklearn.model_selection import train_test_split
import numpy as np

class LeadGenAgent:
    def __init__(self, input_dim=4, hidden_dim=16, num_experts=3):
        self.model = MoEModel(input_dim, hidden_dim, num_experts)
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_model(self, df: pd.DataFrame, epochs=10):
        """
        Train the MoE model on lead data. I'll define a binary label (qualified or not).
        """
        # Basic label: conversion_rate > median => 1, else 0
        df['qualified'] = (df['conversion_rate'] > df['conversion_rate'].median()).astype(int)

        # X: [website_visits, email_clicks, ad_click_through_rate, conversion_rate]
        X = df[['website_visits', 'email_clicks', 'ad_click_through_rate', 'conversion_rate']].values
        y = df['qualified'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            preds = self.model(X_train)
            loss = self.loss_fn(preds, y_train)
            loss.backward()
            self.optimizer.step()

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            test_preds = self.model(X_test)
            test_preds_label = (test_preds > 0.5).float()
            accuracy = (test_preds_label == y_test).float().mean().item()
            print(f"MoE Model Training complete. Test Accuracy: {accuracy:.3f}")

    def score_leads(self, df: pd.DataFrame):
        """
        Use the trained MoE model to produce lead scores in [0,1].
        """
        X = df[['website_visits', 'email_clicks', 'ad_click_through_rate', 'conversion_rate']].values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().numpy()
        df['lead_score'] = preds
        return df

    def recommend_strategy(self, lead_score):
        """
        Simple heuristic: 
         - If lead_score > 0.8 => aggressive multi-channel approach
         - If lead_score > 0.5 => moderate approach
         - else => nurture leads
        """
        if lead_score > 0.8:
            return "Aggressive multi-channel approach with personalized content."
        elif lead_score > 0.5:
            return "Moderate approach, focus on email & social media retargeting."
        else:
            return "Low-intensity nurturing. Automated follow-ups and occasional check-ins."

    def generate_strategies(self, df: pd.DataFrame):
        df['strategy'] = df['lead_score'].apply(self.recommend_strategy)
        return df
