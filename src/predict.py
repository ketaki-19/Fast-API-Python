import joblib
import numpy as np
import os

def predict_data(features):
    """
    Predict diabetes progression using the trained model
    
    Args:
        features: List of 10 feature values in order:
                 [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
    
    Returns:
        Predicted diabetes progression value (float)
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'diabetes_model.pkl')
    
    # Load the model
    model = joblib.load(model_path)
    
    # Convert to numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    return float(prediction[0])