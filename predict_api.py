from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("signal_lamp_predictive_model.pkl")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Reorder columns to match training features
    expected_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    # Perform prediction
    prediction = model.predict(input_df)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    # 👇 Change port to 5000 or any other custom port
    app.run(host='0.0.0.0', port=5000, debug=True)
