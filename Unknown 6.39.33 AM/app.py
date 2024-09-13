from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
import datetime
import sqlite3

app = Flask(__name__)

# Load models and scaler
rf_model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Database connection
def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data.get('radius'), data.get('texture'), data.get('perimeter'), data.get('area')]
    features = np.array([features])
    features_normalized = scaler.transform(features)

    rf_prediction = rf_model.predict(features_normalized)[0]
    rf_class = 'malignant' if rf_prediction == 1 else 'benign'

    # Save prediction to database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO predictions (radius, texture, perimeter, area, prediction, date) VALUES (?, ?, ?, ?, ?, ?)',
                (data.get('radius'), data.get('texture'), data.get('perimeter'), data.get('area'), rf_class, datetime.datetime.now()))
    conn.commit()
    conn.close()

    return jsonify({'rf_prediction': rf_class})

@app.route('/api/past_predictions', methods=['GET'])
def past_predictions():
    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions').fetchall()
    conn.close()

    # Example data for metrics and feature distribution
    metrics = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.93,
        'f1_score': 0.94
    }

    feature_distribution = {
        'labels': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
        'values': [20, 30, 25, 25]
    }

    return jsonify({
        'past_predictions': [dict(row) for row in predictions],
        'metrics': metrics,
        'feature_distribution': feature_distribution
    })

if __name__ == '__main__':
    app.run(debug=True)
