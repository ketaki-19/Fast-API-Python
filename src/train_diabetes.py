from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from data_diabetes import load_data, split_data, get_feature_names  # Changed import

def fit_model(X_train, y_train):
    """
    Train a Random Forest Regressor and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "diabetes_model.pkl")
    print("✓ Model saved as 'diabetes_model.pkl'")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.4f}")
    return mse, r2

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = fit_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save feature names
    feature_names = get_feature_names()
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print("✓ Feature names saved to 'feature_names.json'")
    
    print("\n✅ Training complete!")