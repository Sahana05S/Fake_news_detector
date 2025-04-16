from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load both models
isot_model = joblib.load("Fake_news_detector/model/isot_model.pkl")
liar_model = joblib.load("Fake_news_detector/model/liar_model.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so Streamlit or Chrome Extension can access it

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_choice = data.get('model', 'isot')  # Default to isot if not provided

        if not text:
            return jsonify({'error': 'No input text provided'}), 400

        # Select model
        if model_choice == 'liar':
            prediction = liar_model.predict([text])[0]
        else:
            prediction = isot_model.predict([text])[0]

        label = "Real" if prediction == 1 else "Fake"

        return jsonify({'result': label, 'model_used': model_choice})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
