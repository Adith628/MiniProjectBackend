from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the saved model and scaler
svm = joblib.load('svm_habitability_model.pkl')
scaler = joblib.load('scaler.pkl')

# Column dictionary for feature selection
planetary_stellar_parameter_cols_dict = {   
    "koi_period": "Orbital Period",
    "koi_ror": "Planet-Star Radius Ratio",
    "koi_srho": "Fitted Stellar Density",
    "koi_prad": "Planetary Radius",
    "koi_sma": "Orbit Semi-Major Axis",
    "koi_teq": "Equilibrium Temperature",
    "koi_insol": "Insolation Flux",
    "koi_dor": "Planet-Star Distance over Star Radius",
    "koi_count": "Number of Planet",
    "koi_steff": "Stellar Effective Temperature",
    "koi_slogg": "Stellar Surface Gravity",
    "koi_smet": "Stellar Metallicity",
    "koi_srad": "Stellar Radius",
    "koi_smass": "Stellar Mass"
}

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the CSV file
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Ensure kepoi_name is present
    if 'kepoi_name' not in data.columns:
        return jsonify({"error": "'kepoi_name' column is missing from the input file"}), 400

    kepoi_name = data['kepoi_name']

    # Select specific features
    selected_features = list(planetary_stellar_parameter_cols_dict.keys())
    if not all(col in data.columns for col in selected_features):
        return jsonify({"error": "Input file does not contain all required features"}), 400

    data = data[selected_features]

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Scale the features
    X_predict = scaler.transform(data)

    # Predict using the loaded model
    predictions = svm.predict(X_predict)

    # Prepare the results
    prediction_results = pd.DataFrame({
        'kepoi_name': kepoi_name,
        'habitable': predictions
    })

    # Convert the prediction results to a human-readable format
    prediction_results['habitable'] = prediction_results['habitable'].map({1: 'Habitable', 0: 'Non-Habitable'})

    # Convert the results to JSON
    results = prediction_results.to_dict(orient='records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
