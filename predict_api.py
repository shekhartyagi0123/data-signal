from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("signal_lamp_predictive_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Match training feature columns
    expected_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    # Predict alert using ML model
    alert_prediction = model.predict(input_df)[0]

    # Get failure_count_last_10 from input JSON
    failure_count = data.get("failure_count_last_10", 0)

    # Predictive alert logic
    predictive_alert = 1 if failure_count >= 4 else 0

    # ✅ Return everything including failure count
    return jsonify({
        "prediction": int(alert_prediction),
        "predictive_alert": predictive_alert,
        "failure_count_last_10": failure_count
    })

if __name__ == '__main__':
    # Change port if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
