from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('cancer_model.h5')

# Load the breast cancer dataset for reference
data = load_breast_cancer()
feature_names = data.feature_names

# StandardScaler object (should be the same scaler used when training the model)
scaler = StandardScaler()

# Route for the homepage (input form)
@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

# Route for handling form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data and convert to list of floats
        input_data = [float(request.form[feature]) for feature in feature_names]
        
        # Convert to numpy array and reshape for the model
        input_data_as_array = np.array(input_data).reshape(1, -1)
        
        # Standardize the input data
        input_data_std = scaler.transform(input_data_as_array)
        
        # Make a prediction
        prediction = model.predict(input_data_std)
        prediction_label = np.argmax(prediction)
        
        # Save the input data to a CSV file
        data_file = 'user_input_data.csv'
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Append to the file if it exists, or create a new file
        if os.path.exists(data_file):
            input_df.to_csv(data_file, mode='a', header=False, index=False)
        else:
            input_df.to_csv(data_file, mode='w', header=True, index=False)
        
        # Display result to user
        result = "Malignant" if prediction_label == 0 else "Benign"
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)






# https://induscancer.com/wp-content/uploads/2023/10/What-is-Breast-Cancer-1024x1024.jpg?x60331

# https://drjayanam.com/wp-content/uploads/2023/01/infografic-design-03.webp

# https://assets.delveinsight.com/blog/wp-content/uploads/2018/10/17133520/new-png-info.jpg

# https://actchealth.com/images/Stages-of%20-blood-cancer.jpg

# https://www.check4cancer.com/images/breast-cancer/6_steps_to_performing_a_breast_self_-examination.jpg