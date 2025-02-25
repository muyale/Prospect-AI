# Lead Agent AI

An advanced AI-driven solution that tackles the full spectrum of lead generation challenges—from data generation and predictive scoring to strategy formulation and follow-up automation. This system leverages synthetic business and advertising data, a Chroma vector database, and a Retrieval-Augmented Generation (RAG) pipeline with a Mistral LLM (using Hugging Face) to produce actionable business strategies, follow-up templates, and performance analytics.

## Features

- **Synthetic Data Generation:** Simulate realistic business metrics.
- **Data Processing & Predictive Lead Scoring:** Automate lead scoring using logistic regression.
- **RAG Strategy Generation:** Generate detailed business strategies using expert research loaded into a Chroma vector store.
- **Follow-Up Automation:** Create personalized follow-up email templates.
- **Performance Analytics:** Generate reports and charts to measure lead performance.
- **Interactive Presentation:** Access via Flask API, Streamlit dashboard, and React frontend.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Add your Hugging Face token to the `.env` file.
3. Run the full pipeline: `python src/demo.py`
4. Launch the Streamlit app: `streamlit run streamlit_app.py`
5. (Optional) Start the React app from the `frontend/` folder: `npm start`
