from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the diabetes dataset from sklearn
    
    Returns:
        X: Feature matrix (442 samples, 10 features)
        y: Target values (disease progression)
    """
    print("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Proportion of dataset for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def get_feature_names():
    """
    Get the feature names from the diabetes dataset
    
    Returns:
        List of feature names
    """
    diabetes = load_diabetes()
    return diabetes.feature_names