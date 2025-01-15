from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('customer_segmentation_model.pkl', 'rb') as model_file:
    model_dict = pickle.load(model_file)
    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
        label_encoders = model_dict.get('label_encoders', {})
        target_encoder = model_dict.get('target_encoder', {})
    else:
        raise ValueError("Loaded model is not in the expected format.")

# Preprocess input for demonstration
def preprocess_input(user_input, label_encoders):
    # Transform user input using label encoders
    for column, le in label_encoders.items():
        if column in user_input:
            user_input[column] = le.transform([user_input[column]])[0]
    return user_input

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        processed_input = preprocess_input(user_input, label_encoders)

        # Convert the processed input into a DataFrame for the model
        input_df = pd.DataFrame([processed_input])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)  # Ensure all required columns are present
        prediction = model.predict(input_df)
        predicted_label = target_encoder.inverse_transform(prediction)[0]

        return jsonify(predicted_label=predicted_label)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)