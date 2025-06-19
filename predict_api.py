from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("signal_lamp_predictive_model.pkl")

# Manual threshold limits (from your Signal Lamp form)
MIN_CURRENT = 100
MAX_CURRENT = 150
MIN_VOLTAGE = 100
MAX_VOLTAGE = 120

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Hybrid Rule-Based Check
    if (data["Current_clean"] < MIN_CURRENT or data["Current_clean"] > MAX_CURRENT or
        data["Voltage_clean"] < MIN_VOLTAGE or data["Voltage_clean"] > MAX_VOLTAGE):
        return jsonify({"prediction": 1, "reason": "Threshold breach"})

    # Convert to DataFrame for ML
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # ML Prediction
    prediction = model.predict(input_df)[0]

    return jsonify({"prediction": int(prediction), "reason": "ML model"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
