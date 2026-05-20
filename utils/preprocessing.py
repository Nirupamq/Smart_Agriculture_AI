"""
Data Preprocessing Module
==========================
Handles data loading, cleaning, and preprocessing for the Smart Agriculture AI system.

Author: AI Assistant
Date: May 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'crop_data.csv')

# Feature columns
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Target column
TARGET_COLUMN = 'label'


# ============================================================================
# DATA LOADING
# ============================================================================
def load_crop_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the crop dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file. If None, uses default DATA_PATH.
        
    Returns:
        DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    if file_path is None:
        file_path = DATA_PATH
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'num_samples': len(df),
        'num_features': len(FEATURE_COLUMNS),
        'num_classes': df[TARGET_COLUMN].nunique(),
        'class_distribution': df[TARGET_COLUMN].value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': df[FEATURE_COLUMNS].describe().to_dict()
    }
    return info


# ============================================================================
# DATA CLEANING
# ============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if df_clean.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        for col in FEATURE_COLUMNS:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"✓ Handled missing values")
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    if len(df_clean) < initial_len:
        print(f"✓ Removed {initial_len - len(df_clean)} duplicate rows")
    
    # Clip outliers (optional - based on domain knowledge)
    # For N, P, K: keep values between 0 and 200
    for col in ['N', 'P', 'K']:
        df_clean[col] = df_clean[col].clip(0, 200)
    
    # For temperature: keep between -10 and 50
    df_clean['temperature'] = df_clean['temperature'].clip(-10, 50)
    
    # For humidity: keep between 0 and 100
    df_clean['humidity'] = df_clean['humidity'].clip(0, 100)
    
    # For pH: keep between 0 and 14
    df_clean['ph'] = df_clean['ph'].clip(0, 14)
    
    # For rainfall: keep between 0 and 500
    df_clean['rainfall'] = df_clean['rainfall'].clip(0, 500)
    
    print(f"✓ Data cleaning complete! Shape: {df_clean.shape}")
    return df_clean


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_data(df: pd.DataFrame, fit_encoder: bool = True):
    """
    Preprocess the dataset for machine learning.
    
    Args:
        df: Input DataFrame
        fit_encoder: Whether to fit the label encoder (True for training)
        
    Returns:
        tuple: (X, y, label_encoder, scaler)
            - X: Features array (scaled)
            - y: Encoded labels array
            - label_encoder: Fitted LabelEncoder
            - scaler: Fitted StandardScaler
    """
    # Separate features and target
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Preprocessing complete!")
    print(f"  - Features: {FEATURE_COLUMNS}")
    print(f"  - Number of classes: {len(label_encoder.classes_)}")
    print(f"  - Classes: {list(label_encoder.classes_)}")
    
    return X_scaled, y_encoded, label_encoder, scaler


def preprocess_input(input_data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Preprocess a single input for prediction.
    
    Args:
        input_data: Raw input array
        scaler: Fitted StandardScaler
        
    Returns:
        np.ndarray: Scaled input
    """
    return scaler.transform(input_data)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def get_feature_importance_df(model, feature_names: list = None) -> pd.DataFrame:
    """
    Get feature importance as a DataFrame.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame: Feature importance sorted by importance
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS
    
    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return df_importance


def get_optimal_range(crop: str, df: pd.DataFrame) -> dict:
    """
    Get optimal parameter ranges for a specific crop.
    
    Args:
        crop: Crop name
        df: DataFrame with crop data
        
    Returns:
        dict: Optimal ranges for each feature
    """
    crop_data = df[df[TARGET_COLUMN] == crop][FEATURE_COLUMNS]
    
    if len(crop_data) == 0:
        return None
    
    ranges = {}
    for col in FEATURE_COLUMNS:
        ranges[col] = {
            'min': crop_data[col].min(),
            'max': crop_data[col].max(),
            'mean': crop_data[col].mean(),
            'std': crop_data[col].std()
        }
    
    return ranges


# ============================================================================
# VALIDATION
# ============================================================================
def validate_input(N: float, P: float, K: float, 
                   temperature: float, humidity: float, 
                   ph: float, rainfall: float) -> tuple:
    """
    Validate input parameters.
    
    Args:
        N: Nitrogen content
        P: Phosphorus content
        K: Potassium content
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        ph: pH value
        rainfall: Rainfall in mm
        
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    if not (0 <= N <= 200):
        errors.append("Nitrogen (N) must be between 0 and 200")
    
    if not (0 <= P <= 200):
        errors.append("Phosphorus (P) must be between 0 and 200")
    
    if not (0 <= K <= 200):
        errors.append("Potassium (K) must be between 0 and 200")
    
    if not (-10 <= temperature <= 50):
        errors.append("Temperature must be between -10 and 50°C")
    
    if not (0 <= humidity <= 100):
        errors.append("Humidity must be between 0 and 100%")
    
    if not (0 <= ph <= 14):
        errors.append("pH must be between 0 and 14")
    
    if not (0 <= rainfall <= 500):
        errors.append("Rainfall must be between 0 and 500mm")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Test the preprocessing module
    print("=" * 60)
    print("TESTING PREPROCESSING MODULE")
    print("=" * 60)
    
    # Load data
    df = load_crop_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get info
    info = get_data_info(df)
    print(f"\nNumber of samples: {info['num_samples']}")
    print(f"Number of features: {info['num_features']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Classes: {list(info['class_distribution'].keys())}")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Preprocess
    X, y, label_encoder, scaler = preprocess_data(df_clean)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    print("\n✓ Preprocessing module test complete!")