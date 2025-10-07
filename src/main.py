from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from predict import predict_data
import json
import os

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes progression using Machine Learning",
    version="1.0.0"
)

# Load feature names
try:
    feature_path = os.path.join(os.path.dirname(__file__), 'feature_names.json')
    with open(feature_path, 'r') as f:
        FEATURE_NAMES = json.load(f)
except:
    FEATURE_NAMES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

class DiabetesFeatures(BaseModel):
    """Input features for diabetes prediction"""
    age: float = Field(..., description="Age (standardized)")
    sex: float = Field(..., description="Sex (standardized)")
    bmi: float = Field(..., description="Body mass index (standardized)")
    bp: float = Field(..., description="Average blood pressure (standardized)")
    s1: float = Field(..., description="tc, total serum cholesterol")
    s2: float = Field(..., description="ldl, low-density lipoproteins")
    s3: float = Field(..., description="hdl, high-density lipoproteins")
    s4: float = Field(..., description="tch, total cholesterol / HDL")
    s5: float = Field(..., description="ltg, log of serum triglycerides")
    s6: float = Field(..., description="glu, blood sugar level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,
                "s1": -0.04,
                "s2": -0.03,
                "s3": -0.00,
                "s4": -0.00,
                "s5": 0.00,
                "s6": -0.03
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    interpretation: str
    features: dict

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Diabetes Progression Prediction API",
        "status": "healthy",
        "model": "Random Forest Regressor",
        "dataset": "Diabetes Dataset",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "features": "/features"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesFeatures):
    """
    Predict diabetes progression based on patient features.
    
    All features should be standardized values (typically ranging from -0.2 to 0.2).
    Returns a quantitative measure of disease progression one year after baseline.
    """
    try:
        # Convert to list in the correct order
        features = [
            data.age, data.sex, data.bmi, data.bp,
            data.s1, data.s2, data.s3, data.s4, data.s5, data.s6
        ]
        
        # Get prediction
        prediction = predict_data(features)
        
        # Get interpretation
        interpretation = get_interpretation(prediction)
        
        return PredictionResponse(
            prediction=round(prediction, 2),
            interpretation=interpretation,
            features=data.dict()
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="Model file not found. Please train the model first by running 'python train.py'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def model_info():
    """Get information about the ML model"""
    return {
        "model_type": "Random Forest Regressor",
        "dataset": "Diabetes Dataset (sklearn)",
        "target": "Diabetes progression after one year",
        "n_features": 10,
        "feature_names": FEATURE_NAMES,
        "description": "Predicts quantitative measure of disease progression one year after baseline",
        "note": "All features are standardized (mean-centered and scaled)"
    }

@app.get("/features")
def get_features():
    """Get list of features required for prediction"""
    return {
        "features": FEATURE_NAMES,
        "description": "All features are standardized values",
        "typical_range": "Values typically range from -0.2 to 0.2",
        "count": len(FEATURE_NAMES)
    }

def get_interpretation(prediction: float) -> str:
    """
    Provide interpretation of prediction value
    
    Args:
        prediction: Predicted disease progression value
    
    Returns:
        Interpretation string
    """
    if prediction < 75:
        return "Low disease progression"
    elif prediction < 150:
        return "Moderate disease progression"
    elif prediction < 225:
        return "High disease progression"
    else:
        return "Very high disease progression"