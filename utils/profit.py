"""
Profit Intelligence Module
==========================
Handles crop profitability analysis and economic recommendations.

Author: AI Assistant
Date: May 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# CROP PROFIT DATA
# ============================================================================
# Realistic crop data with yield and market prices
# Yield in quintals/hectare, Price in INR/quintal
CROP_DATA = {
    'rice': {
        'yield_per_hectare': 50,  # quintals
        'market_price': 2200,  # INR per quintal
        'cost_of_cultivation': 35000,  # INR per hectare
        'duration_months': 6,
        'water_requirement': 'high',
        'soil_type': 'alluvial',
        'season': 'kharif',
        'description': 'Grown in flooded fields, requires high moisture'
    },
    'wheat': {
        'yield_per_hectare': 45,
        'market_price': 2400,
        'cost_of_cultivation': 30000,
        'duration_months': 5,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'rabi',
        'description': 'Rabi crop, needs cool climate'
    },
    'maize': {
        'yield_per_hectare': 60,
        'market_price': 1900,
        'cost_of_cultivation': 28000,
        'duration_months': 4,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'kharif',
        'description': 'Versatile crop, used for food and feed'
    },
    'cotton': {
        'yield_per_hectare': 15,
        'market_price': 6500,
        'cost_of_cultivation': 45000,
        'duration_months': 8,
        'water_requirement': 'medium',
        'soil_type': 'black',
        'season': 'kharif',
        'description': 'Cash crop, requires warm climate'
    },
    'sugarcane': {
        'yield_per_hectare': 750,  # quintals
        'market_price': 350,
        'cost_of_cultivation': 60000,
        'duration_months': 12,
        'water_requirement': 'high',
        'soil_type': 'alluvial',
        'season': 'annual',
        'description': 'Long duration crop, high water need'
    },
    'coffee': {
        'yield_per_hectare': 8,
        'market_price': 15000,
        'cost_of_cultivation': 80000,
        'duration_months': 24,
        'water_requirement': 'medium',
        'soil_type': 'red',
        'season': 'perennial',
        'description': 'Hill crop, shade-loving'
    },
    'groundnuts': {
        'yield_per_hectare': 25,
        'market_price': 5500,
        'cost_of_cultivation': 25000,
        'duration_months': 4,
        'water_requirement': 'low',
        'soil_type': 'sandy',
        'season': 'kharif',
        'description': 'Oilseed, nitrogen-fixing'
    },
    'potato': {
        'yield_per_hectare': 200,
        'market_price': 1200,
        'cost_of_cultivation': 40000,
        'duration_months': 4,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'rabi',
        'description': 'High yield, popular vegetable'
    },
    'tomato': {
        'yield_per_hectare': 150,
        'market_price': 1500,
        'cost_of_cultivation': 35000,
        'duration_months': 4,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'all',
        'description': 'High value vegetable'
    },
    'soybean': {
        'yield_per_hectare': 20,
        'market_price': 4500,
        'cost_of_cultivation': 22000,
        'duration_months': 4,
        'water_requirement': 'low',
        'soil_type': 'black',
        'season': 'kharif',
        'description': 'Protein-rich oilseed'
    },
    'pomegranate': {
        'yield_per_hectare': 150,
        'market_price': 4000,
        'cost_of_cultivation': 70000,
        'duration_months': 18,
        'water_requirement': 'low',
        'soil_type': 'all',
        'season': 'perennial',
        'description': 'Fruit crop, drought tolerant'
    },
    'millets': {
        'yield_per_hectare': 30,
        'market_price': 3000,
        'cost_of_cultivation': 15000,
        'duration_months': 3,
        'water_requirement': 'low',
        'soil_type': 'light',
        'season': 'kharif',
        'description': 'Drought resistant, nutritious'
    },
    'banana': {
        'yield_per_hectare': 300,
        'market_price': 800,
        'cost_of_cultivation': 50000,
        'duration_months': 12,
        'water_requirement': 'high',
        'soil_type': 'alluvial',
        'season': 'all',
        'description': 'High yield fruit crop'
    },
    'chilli': {
        'yield_per_hectare': 30,
        'market_price': 8000,
        'cost_of_cultivation': 30000,
        'duration_months': 5,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'all',
        'description': 'Spice crop, high value'
    },
    'onion': {
        'yield_per_hectare': 150,
        'market_price': 1500,
        'cost_of_cultivation': 25000,
        'duration_months': 4,
        'water_requirement': 'low',
        'soil_type': 'sandy',
        'season': 'rabi',
        'description': 'Essential vegetable, good storage'
    },
    'brinjal': {
        'yield_per_hectare': 200,
        'market_price': 1200,
        'cost_of_cultivation': 20000,
        'duration_months': 4,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'all',
        'description': 'Popular vegetable'
    },
    'cabbage': {
        'yield_per_hectare': 250,
        'market_price': 800,
        'cost_of_cultivation': 18000,
        'duration_months': 3,
        'water_requirement': 'medium',
        'soil_type': 'loamy',
        'season': 'rabi',
        'description': 'Leafy vegetable'
    },
    'moong': {
        'yield_per_hectare': 15,
        'market_price': 7000,
        'cost_of_cultivation': 18000,
        'duration_months': 3,
        'water_requirement': 'low',
        'soil_type': 'all',
        'season': 'kharif',
        'description': 'Pulses, nitrogen-fixing'
    },
    'sunflower': {
        'yield_per_hectare': 15,
        'market_price': 5500,
        'cost_of_cultivation': 20000,
        'duration_months': 4,
        'water_requirement': 'low',
        'soil_type': 'black',
        'season': 'kharif',
        'description': 'Oilseed, drought tolerant'
    }
}


# ============================================================================
# PROFIT CALCULATION FUNCTIONS
# ============================================================================
def calculate_profit(crop: str) -> Dict:
    """
    Calculate profit metrics for a crop.
    
    Args:
        crop: Crop name
        
    Returns:
        dict: Profit metrics including gross and net profit
    """
    crop_lower = crop.lower().strip()
    
    if crop_lower not in CROP_DATA:
        return None
    
    data = CROP_DATA[crop_lower]
    
    # Calculate gross income
    gross_income = data['yield_per_hectare'] * data['market_price']
    
    # Calculate net profit
    net_profit = gross_income - data['cost_of_cultivation']
    
    # Calculate ROI
    roi = (net_profit / data['cost_of_cultivation']) * 100
    
    return {
        'crop': crop_lower.capitalize(),
        'gross_income': gross_income,
        'cost_of_cultivation': data['cost_of_cultivation'],
        'net_profit': net_profit,
        'roi': roi,
        'yield_per_hectare': data['yield_per_hectare'],
        'market_price': data['market_price'],
        'duration_months': data['duration_months'],
        'water_requirement': data['water_requirement'],
        'season': data['season']
    }


def get_all_profits() -> pd.DataFrame:
    """
    Get profit data for all crops as a DataFrame.
    
    Returns:
        DataFrame: Profit data for all crops
    """
    profits = []
    for crop in CROP_DATA.keys():
        profit_data = calculate_profit(crop)
        if profit_data:
            profits.append(profit_data)
    
    df = pd.DataFrame(profits)
    df = df.sort_values('net_profit', ascending=False)
    return df


def get_most_profitable_crop() -> Dict:
    """
    Get the most profitable crop.
    
    Returns:
        dict: Most profitable crop data
    """
    df = get_all_profits()
    if len(df) > 0:
        return df.iloc[0].to_dict()
    return None


def get_top_n_profitable(n: int = 5) -> List[Dict]:
    """
    Get top N most profitable crops.
    
    Args:
        n: Number of crops to return
        
    Returns:
        list: Top N profitable crops
    """
    df = get_all_profits()
    return df.head(n).to_dict('records')


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================
def compare_crops(crops: List[str]) -> pd.DataFrame:
    """
    Compare profit metrics for multiple crops.
    
    Args:
        crops: List of crop names
        
    Returns:
        DataFrame: Comparison of crops
    """
    comparison = []
    for crop in crops:
        profit_data = calculate_profit(crop)
        if profit_data:
            comparison.append(profit_data)
    
    return pd.DataFrame(comparison)


def get_profit_by_water_requirement(water_req: str) -> List[Dict]:
    """
    Get crops filtered by water requirement.
    
    Args:
        water_req: Water requirement ('low', 'medium', 'high')
        
    Returns:
        list: Crops with specified water requirement
    """
    results = []
    for crop, data in CROP_DATA.items():
        if data['water_requirement'] == water_req.lower():
            profit_data = calculate_profit(crop)
            results.append(profit_data)
    
    return sorted(results, key=lambda x: x['net_profit'], reverse=True)


def get_profit_by_season(season: str) -> List[Dict]:
    """
    Get crops filtered by season.
    
    Args:
        season: Season ('kharif', 'rabi', 'all', 'perennial', 'annual')
        
    Returns:
        list: Crops for specified season
    """
    results = []
    for crop, data in CROP_DATA.items():
        if data['season'] == season.lower():
            profit_data = calculate_profit(crop)
            results.append(profit_data)
    
    return sorted(results, key=lambda x: x['net_profit'], reverse=True)


# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================
def recommend_by_profit_and_conditions(
    predicted_crops: List[Tuple[str, float]], 
    rainfall: float = None,
    humidity: float = None,
    temperature: float = None
) -> Dict:
    """
    Recommend the most profitable crop from predicted crops considering conditions.
    
    Args:
        predicted_crops: List of (crop_name, confidence) tuples
        rainfall: Rainfall in mm (optional)
        humidity: Humidity percentage (optional)
        temperature: Temperature in Celsius (optional)
        
    Returns:
        dict: Recommendation with AI prediction and profit analysis
    """
    results = {
        'ai_prediction': None,
        'most_profitable': None,
        'comparison': []
    }
    
    # Get AI prediction (highest confidence)
    if predicted_crops:
        results['ai_prediction'] = {
            'crop': predicted_crops[0][0],
            'confidence': predicted_crops[0][1]
        }
    
    # Get profit data for predicted crops
    crop_profits = []
    for crop, confidence in predicted_crops:
        profit_data = calculate_profit(crop)
        if profit_data:
            profit_data['ai_confidence'] = confidence
            crop_profits.append(profit_data)
    
    # Sort by profit
    crop_profits = sorted(crop_profits, key=lambda x: x['net_profit'], reverse=True)
    
    # Get most profitable
    if crop_profits:
        results['most_profitable'] = {
            'crop': crop_profits[0]['crop'],
            'profit': crop_profits[0]['net_profit']
        }
    
    results['comparison'] = crop_profits
    
    return results


def get_crop_suitability(crop: str, conditions: Dict) -> Dict:
    """
    Analyze crop suitability based on conditions.
    
    Args:
        crop: Crop name
        conditions: Dictionary with environmental conditions
        
    Returns:
        dict: Suitability analysis
    """
    crop_lower = crop.lower().strip()
    
    if crop_lower not in CROP_DATA:
        return None
    
    data = CROP_DATA[crop_lower]
    score = 0
    reasons = []
    
    # Check rainfall
    if 'rainfall' in conditions:
        if data['water_requirement'] == 'high' and conditions['rainfall'] > 200:
            score += 25
            reasons.append("✓ High rainfall suits this water-intensive crop")
        elif data['water_requirement'] == 'low' and conditions['rainfall'] < 100:
            score += 25
            reasons.append("✓ Low rainfall suits this drought-tolerant crop")
        elif data['water_requirement'] == 'medium':
            score += 20
            reasons.append("✓ Moderate rainfall suitable")
        else:
            score += 10
            reasons.append("⚠ Rainfall may not be optimal")
    
    # Check humidity
    if 'humidity' in conditions:
        if data['water_requirement'] == 'high' and conditions['humidity'] > 70:
            score += 25
            reasons.append("✓ High humidity beneficial")
        elif data['water_requirement'] == 'low' and conditions['humidity'] < 50:
            score += 25
            reasons.append("✓ Low humidity suitable for drought-tolerant crop")
        else:
            score += 15
            reasons.append("⚠ Humidity may not be ideal")
    
    # Check temperature (simplified)
    if 'temperature' in conditions:
        temp = conditions['temperature']
        if 15 <= temp <= 35:
            score += 25
            reasons.append("✓ Temperature in suitable range")
        else:
            score += 10
            reasons.append("⚠ Temperature may affect growth")
    
    # Check pH
    if 'ph' in conditions:
        ph = conditions['ph']
        if 5.5 <= ph <= 7.5:
            score += 25
            reasons.append("✓ Soil pH optimal for this crop")
        elif 5.0 <= ph <= 8.0:
            score += 15
            reasons.append("⚠ pH slightly outside ideal range")
        else:
            score += 5
            reasons.append("⚠ pH may limit crop performance")
    
    return {
        'crop': crop,
        'suitability_score': min(score, 100),
        'reasons': reasons,
        'water_requirement': data['water_requirement'],
        'season': data['season'],
        'duration_months': data['duration_months']
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Test the profit module
    print("=" * 60)
    print("TESTING PROFIT MODULE")
    print("=" * 60)
    
    # Get all profits
    print("\n📊 Profit Analysis for All Crops:")
    print("-" * 60)
    
    df = get_all_profits()
    print(df[['crop', 'net_profit', 'roi', 'water_requirement', 'season']].to_string(index=False))
    
    # Get most profitable
    print("\n\n💰 Most Profitable Crop:")
    most_profitable = get_most_profitable_crop()
    print(f"  Crop: {most_profitable['crop']}")
    print(f"  Net Profit: ₹{most_profitable['net_profit']:,}/hectare")
    print(f"  ROI: {most_profitable['roi']:.1f}%")
    
    # Test comparison
    print("\n\n🔄 Crop Comparison (rice, wheat, cotton):")
    comparison = compare_crops(['rice', 'wheat', 'cotton'])
    print(comparison[['crop', 'net_profit', 'roi']].to_string(index=False))
    
    print("\n✓ Profit module test complete!")