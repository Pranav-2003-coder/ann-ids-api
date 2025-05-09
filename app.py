from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained ANN model and label encoder
model = joblib.load("ann_model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "✅ ANN Intrusion Detection System is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    df = pd.read_csv(request.files['file'])
    if 'label' in df.columns:
        df = df.drop('label', axis=1)
    preds = model.predict(df)
    decoded = encoder.inverse_transform(preds)
    return jsonify({'predictions': decoded.tolist()})

# Correct Render port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamically provides this env var
    app.run(host="0.0.0.0", port=port)
