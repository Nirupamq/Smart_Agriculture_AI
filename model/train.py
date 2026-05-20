"""
Model Training Module
====================
Handles machine learning model training, evaluation, and persistence.

Author: AI Assistant
Date: May 2026
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (
    load_crop_data, 
    clean_data, 
    preprocess_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pkl')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_model(
    n_estimators: int = 200,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5
) -> dict:
    """
    Train a RandomForestClassifier on the crop dataset.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at leaf node
        random_state: Random state for reproducibility
        test_size: Proportion of data for testing
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict: Training results including model, metrics, and files
    """
    print("=" * 60)
    print("SMART AGRICULTURE - MODEL TRAINING")
    print("=" * 60)
    
    # Load and clean data
    print("\n[1/6] Loading dataset...")
    df = load_crop_data()
    df_clean = clean_data(df)
    
    # Preprocess data
    print("\n[2/6] Preprocessing data...")
    X, y, label_encoder, scaler = preprocess_data(df_clean)
    
    # Split data
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Initialize model
    print("\n[4/6] Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Train model
    model.fit(X_train, y_train)
    print("  ✓ Model trained successfully!")
    
    # Evaluate model
    print("\n[5/6] Evaluating model...")
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Print results
    print(f"\n  📊 Model Performance:")
    print(f"  ─────────────────────────────────────")
    print(f"  Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"  Testing Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Cross-Val Score:   {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    print(f"\n  📈 Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"    {row['Feature']:12s}: {'█' * int(row['Importance']*50):50s} {row['Importance']:.3f}")
    
    # Save model
    print("\n[6/6] Saving model...")
    save_model(model, label_encoder, scaler)
    
    results = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'feature_importance': feature_importance,
        'n_samples': len(df_clean),
        'n_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_)
    }
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    return results


def save_model(model, label_encoder, scaler):
    """
    Save the trained model and associated objects.
    
    Args:
        model: Trained model
        label_encoder: Fitted LabelEncoder
        scaler: Fitted StandardScaler
    """
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to: {MODEL_FILE}")
    
    # Save label encoder
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Label encoder saved to: {ENCODER_FILE}")
    
    # Save scaler
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved to: {SCALER_FILE}")


def load_model():
    """
    Load the trained model and associated objects.
    
    Returns:
        dict: Dictionary with model, label_encoder, and scaler
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    
    # Load model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ Model loaded from: {MODEL_FILE}")
    
    # Load label encoder
    with open(ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"  ✓ Label encoder loaded from: {ENCODER_FILE}")
    
    # Load scaler
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Scaler loaded from: {SCALER_FILE}")
    
    return {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler
    }


def is_model_trained() -> bool:
    """
    Check if a trained model exists.
    
    Returns:
        bool: True if model exists, False otherwise
    """
    return os.path.exists(MODEL_FILE)


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
def tune_hyperparameters(df: pd.DataFrame = None) -> dict:
    """
    Tune hyperparameters using GridSearchCV.
    
    Args:
        df: DataFrame with crop data (optional, will load if not provided)
        
    Returns:
        dict: Best parameters and scores
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        df = load_crop_data()
        df = clean_data(df)
    
    # Preprocess
    X, y, label_encoder, scaler = preprocess_data(df)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search
    print("\nRunning GridSearchCV (this may take a few minutes)...")
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n  Best Parameters: {grid_search.best_params_}")
    print(f"  Best Score: {grid_search.best_score_*100:.2f}%")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test, label_encoder) -> dict:
    """
    Evaluate the model with detailed metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted LabelEncoder
        
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        roc_auc = None
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'class_names': class_names
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Check if model exists
    if is_model_trained():
        print("\n⚠ Model already exists. Loading existing model...")
        model_data = load_model()
        print(f"\n✓ Model loaded successfully!")
        print(f"  Classes: {list(model_data['label_encoder'].classes_)}")
    else:
        print("\n⚠ No trained model found. Training new model...")
        results = train_model()
        print(f"\n✓ Model trained and saved successfully!")
        print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"  Classes: {results['classes']}")