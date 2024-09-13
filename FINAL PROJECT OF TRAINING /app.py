from flask import Flask , render_template , request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model 
model = tf.keras.models.load_model('breast_cancer_model.h5') # Load your trained model

# Initialize the Flask app
app = Flask(__name__)

# Assume you have a fitted StandardScalar (load it if Necessary)
scalar = StandardScaler() # Replace with actual trained scalar 

@app.route('/')
def home():
    return render_template('Homepage.html')

@app.route('/predict',methods=['POST'])
def predict():
    try: 
        # Get the input data from the form 
        input_data = [float(x) for x in request.form.values()]

        # Convert the numpy array and reshape it 
        input_data_as_numpy_array = np.asarray(input_data).reshape(1.-1)

        # Standardize the input 
        input_data_std = scalar.transform(input_data_as_numpy_array)

        # Make a prediction
        prediction = model.predict(input_data_std)
        prediction_label = np.argmax(prediction)

        # Display the result
        if prediction_label == 0:
            result = " The Tumor is malignant "
        else:
            result = " The Tumor is Benign "

    except ValueError as e :
        # Handle inavlid input or any other errors
        result = f"Error: {e}"

    return render_template('Homepage.html',prediction_text = result)

if __name__ =="__main__":
    app.run(debug = True)