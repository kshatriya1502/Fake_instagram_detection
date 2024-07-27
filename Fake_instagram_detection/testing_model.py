import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def predict_fake_account(features):
    # Convert features to DataFrame to match training data structure
    feature_names = ['profile pic', 'nums/length username', 'fullname words',
       'nums/length fullname', 'name==username', 'description length',
       'external URL', 'private', '#posts', '#followers', '#follows']
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Load the scaler and transform the input data
    scaler_x = joblib.load('scaler_x.pkl')
    input_data = scaler_x.transform(input_data)
    
    # Load the trained model
    model = tf.keras.models.load_model('instagram_fake_detector_model.h5')
    
    # Predict the class (fake or not)
    predicted = model.predict(input_data)
    predicted_class = np.argmax(predicted, axis=1)
    
    return 'Fake Account' if predicted_class[0] == 1 else 'Real Account'

# Example usage:
example_features = [ 1 , 0.27 , 0 , 0 , 0 , 53 ,	 0	 , 0	, 32 ,	 1000	, 955 ]
result = predict_fake_account(example_features)
print(result)
