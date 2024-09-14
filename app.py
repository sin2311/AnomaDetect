#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle


# In[4]:


# Load the dataset
df=pd.read_excel(r"C:\Users\ssindhu\Downloads\AnomaData.xlsx")


# In[5]:


# Split the data into features (X) and target variable (y)
X = df.drop(['time', 'y'], axis=1)  # Assuming 'time' is not a feature for prediction
y = df['y']


# In[6]:


# Train a logistic regression model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_scaled, y)


# In[7]:


# Serialize the trained model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(logistic_regression, f)


# In[8]:


# Initialize Flask application
app = Flask(__name__)


# In[9]:


# Load the trained model
# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[10]:


import tempfile
import numpy as np
import json
from flask import Flask, request, send_file
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('logistic_regression_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/predict_new_one', methods=['POST'])
def predict_new_one():
    try:
        # Get input data from request (ensure it's in JSON format)
        df = request.get_json()

        # Convert input data to numpy array (assuming 'features' is a list of lists)
        features = np.array(df['features'])

        # Standardize the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Make predictions using the loaded model
        predictions = model.predict(features_scaled)

        # Prepare the predictions as a JSON response
        response = {'predictions': predictions.tolist()}

        # Create a temporary file to store the JSON data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        with open(temp_file.name, 'w') as file:
            json.dump(response, file)

        # Close the temporary file
        temp_file.close()

        # Return the file for download as a JSON file
        return send_file(temp_file.name, as_attachment=True, attachment_filename='predictions.json')

    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True)

