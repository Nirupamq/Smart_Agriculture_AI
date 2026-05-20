"""
Prediction Module
=================
Handles crop prediction with probability estimates and profit analysis.

Author: AI Assistant
Date: May 2026
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train import load_model, is_model_trained
from utils.profit import (
    calculate_profit, 
    get_all_profits, 
    recommend_by_profit_and_conditions,
    get_crop_suitability
)
from utils.preprocessing import FEATURE_COLUMNS


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================
def predict_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    top_n: int = 3
) -> Dict:
    """
    Predict the best crop based on input parameters.
    
    Args:
        N: Nitrogen content (0-200)
        P: Phosphorus content (0-200)
        K: Potassium content (0-200)
        temperature: Temperature in Celsius (-10 to 50)
        humidity: Humidity percentage (0-100)
        ph: Soil pH value (0-14)
        rainfall: Rainfall in mm (0-500)
        top_n: Number of top predictions to return
        
    Returns:
        dict: Prediction results with crops and confidence scores
    """
    # Check if model is trained
    if not is_model_trained():
        raise FileNotFoundError("Model not trained. Please run model/train.py first.")
    
    # Load model
    model_data = load_model()
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    scaler = model_data['scaler']
    
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Get predictions
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    # Get top N predictions
    top_indices = np.argsort(probabilities[0])[::-1][:top_n]
    
    # Build results
    predicted_crops = []
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[0][idx] * 100
        predicted_crops.append((crop_name, confidence))
    
    # Get best crop
    best_crop = predicted_crops[0][0]
    best_confidence = predicted_crops[0][1]
    
    # Get profit data
    profit_data = calculate_profit(best_crop)
    
    # Get profit comparison for all predicted crops
    profit_comparison = []
    for crop, confidence in predicted_crops:
        profit = calculate_profit(crop)
        if profit:
            profit['ai_confidence'] = confidence
            profit_comparison.append(profit)
    
    # Sort by profit
    profit_comparison = sorted(profit_comparison, key=lambda x: x['net_profit'], reverse=True)
    
    # Get most profitable from predicted
    most_profitable = profit_comparison[0] if profit_comparison else None
    
    # Get suitability analysis
    conditions = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    suitability = get_crop_suitability(best_crop, conditions)
    
    # Build final result
    result = {
        'success': True,
        'input': {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        },
        'ai_prediction': {
            'best_crop': best_crop,
            'confidence': best_confidence,
            'top_crops': predicted_crops
        },
        'profit_analysis': {
            'most_profitable': most_profitable['crop'] if most_profitable else None,
            'most_profitable_profit': most_profitable['net_profit'] if most_profitable else None,
            'comparison': profit_comparison
        },
        'suitability': suitability,
        'conditions': conditions
    }
    
    return result


def predict_crop_simple(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> Tuple[str, float]:
    """
    Simple prediction function returning just crop and confidence.
    
    Args:
        N: Nitrogen content
        P: Phosphorus content
        K: Potassium content
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        ph: Soil pH value
        rainfall: Rainfall in mm
        
    Returns:
        tuple: (crop_name, confidence)
    """
    result = predict_crop(N, P, K, temperature, humidity, ph, rainfall, top_n=1)
    return result['ai_prediction']['best_crop'], result['ai_prediction']['confidence']


# ============================================================================
# EXPLAINABLE AI FUNCTIONS
# ============================================================================
def explain_prediction(result: Dict) -> str:
    """
    Generate human-readable explanation for the prediction.
    
    Args:
        result: Prediction result from predict_crop
        
    Returns:
        str: Explanation text
    """
    best_crop = result['ai_prediction']['best_crop']
    confidence = result['ai_prediction']['confidence']
    conditions = result['conditions']
    suitability = result['suitability']
    
    explanation = f"""
🌾 **Why {best_crop.capitalize()}?**

Based on your input conditions:
- Nitrogen (N): {conditions['N']}
- Phosphorus (P): {conditions['P']}
- Potassium (K): {conditions['K']}
- Temperature: {conditions['temperature']}°C
- Humidity: {conditions['humidity']}%
- Soil pH: {conditions['ph']}
- Rainfall: {conditions['rainfall']}mm

"""
    
    # Add suitability reasons
    if suitability and 'reasons' in suitability:
        explanation += "**Suitability Analysis:**\n"
        for reason in suitability['reasons']:
            explanation += f"- {reason}\n"
    
    # Add confidence info
    explanation += f"""
**AI Confidence:** {confidence:.1f}%

This crop is recommended because the environmental conditions 
match its optimal growing requirements.
"""
    
    return explanation


def get_feature_impact(N: float, P: float, K: float, temperature: float,
                       humidity: float, ph: float, rainfall: float) -> Dict:
    """
    Analyze which features had the most impact on the prediction.
    
    Args:
        input parameters
        
    Returns:
        dict: Feature impact analysis
    """
    # Load model
    model_data = load_model()
    model = model_data['model']
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create feature values dict
    values = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    # Calculate impact (importance * normalized value)
    impacts = []
    for i, feature in enumerate(FEATURE_COLUMNS):
        # Normalize value (approximate ranges)
        if feature == 'N':
            normalized = values[feature] / 100
        elif feature == 'P':
            normalized = values[feature] / 100
        elif feature == 'K':
            normalized = values[feature] / 100
        elif feature == 'temperature':
            normalized = values[feature] / 30
        elif feature == 'humidity':
            normalized = values[feature] / 80
        elif feature == 'ph':
            normalized = values[feature] / 7
        elif feature == 'rainfall':
            normalized = values[feature] / 200
        else:
            normalized = 0.5
        
        impact = importance[i] * normalized
        impacts.append({
            'feature': feature,
            'importance': importance[i],
            'value': values[feature],
            'impact': impact
        })
    
    # Sort by impact
    impacts = sorted(impacts, key=lambda x: x['impact'], reverse=True)
    
    return {
        'feature_importance': impacts,
        'most_impactful': impacts[0]['feature'] if impacts else None
    }


# ============================================================================
# BATCH PREDICTION
# ============================================================================
def predict_batch(inputs: List[Dict]) -> List[Dict]:
    """
    Make predictions for multiple inputs.
    
    Args:
        inputs: List of input dictionaries
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    for input_data in inputs:
        try:
            result = predict_crop(
                N=input_data.get('N', 0),
                P=input_data.get('P', 0),
                K=input_data.get('K', 0),
                temperature=input_data.get('temperature', 25),
                humidity=input_data.get('humidity', 60),
                ph=input_data.get('ph', 6.5),
                rainfall=input_data.get('rainfall', 150)
            )
            results.append(result)
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e)
            })
    
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Test the prediction module
    print("=" * 60)
    print("TESTING PREDICTION MODULE")
    print("=" * 60)
    
    # Check if model is trained
    if not is_model_trained():
        print("\n⚠ Model not found. Please train the model first.")
        print("Run: python model/train.py")
    else:
        # Test prediction
        print("\n🔮 Testing prediction with sample data...")
        print("Input: N=90, P=40, K=40, Temp=20, Humidity=82, pH=6.5, Rainfall=200")
        
        result = predict_crop(
            N=90, P=40, K=40,
            temperature=20, humidity=82,
            ph=6.5, rainfall=200,
            top_n=3
        )
        
        print(f"\n✅ Prediction Result:")
        print(f"  Best Crop: {result['ai_prediction']['best_crop']}")
        print(f"  Confidence: {result['ai_prediction']['confidence']:.1f}%")
        
        print(f"\n  Top 3 Predictions:")
        for crop, conf in result['ai_prediction']['top_crops']:
            print(f"    {crop}: {conf:.1f}%")
        
        print(f"\n  💰 Profit Analysis:")
        print(f"    Most Profitable: {result['profit_analysis']['most_profitable']}")
        print(f"    Profit: ₹{result['profit_analysis']['most_profitable_profit']:,}/hectare")
        
        print(f"\n  📊 Suitability Score: {result['suitability']['suitability_score']}%")
        
        # Test explanation
        print(f"\n  💡 Explanation:")
        explanation = explain_prediction(result)
        print(explanation)
        
        print("\n✓ Prediction module test complete!")