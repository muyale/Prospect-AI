from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/api/leads', methods=['GET'])
def get_leads():
    df = pd.read_csv("data/processed/scored_data.csv")
    leads = df.to_dict(orient='records')
    return jsonify(leads)

@app.route('/api/report', methods=['GET'])
def get_report():
    try:
        with open("data/final/leads_report.txt", "r") as f:
            content = f.read()
        return jsonify({"report": content})
    except FileNotFoundError:
        return jsonify({"report": "No report found. Please run the pipeline."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
