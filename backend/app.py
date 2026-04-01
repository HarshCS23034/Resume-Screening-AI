from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_decision, train_model
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({'status': 'online', 'message': 'Resume Screening API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction = predict_decision(data)
        return jsonify({'status': 'success', 'prediction': prediction})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train', methods=['GET'])
def train():
    try:
        train_model()
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'model.pkl')):
        print("Model file not found, training...")
        train_model()

    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
