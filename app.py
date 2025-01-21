from flask import Flask, render_template, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import logging
import numpy as np

app = Flask(__name__)

model_path = "./customer_segmentation_catboost.cbm"

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    model = CatBoostClassifier()
    model.load_model(model_path)
    logger.info("Model loaded successfully")
    return model

# Preprocess input for demonstration
def preprocess_input(user_input):
    # Assuming no label encoding is needed for CatBoost
    return user_input

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()
        user_input = request.get_json()
        logger.info(f"Received input: {user_input}")
        processed_input = preprocess_input(user_input)

        # Convert the processed input into a DataFrame for the model
        input_df = pd.DataFrame([processed_input])
        input_df = input_df.reindex(columns=model.feature_names_, fill_value=0)  # Ensure all required columns are present
        prediction = model.predict(input_df)

        predicted_label = prediction[0]
        logger.info(f"Predicted label: {predicted_label}")

        # Convert the predicted label to a string if it's an ndarray
        if isinstance(predicted_label, (np.ndarray, list)):
            predicted_label = predicted_label.tolist() if isinstance(predicted_label, np.ndarray) else predicted_label
            predicted_label = predicted_label[0] if isinstance(predicted_label, list) else predicted_label

        return jsonify(predicted_label=predicted_label)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify(error=str(e))

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting app on port {port}")
    app.run(port=port, debug=True)