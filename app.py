"""
Smart Agriculture AI Advisor
=============================
A premium, production-quality AI-powered crop recommendation system.
Demo-ready with clean UI, strong insights, and smooth UX.

Author: AI Assistant
Date: May 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.train import is_model_trained, train_model, load_model
from model.predict import predict_crop, explain_prediction, get_feature_impact
from utils.profit import get_all_profits, calculate_profit
from utils.preprocessing import load_crop_data, FEATURE_COLUMNS


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Smart Agriculture AI Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS - Premium Styling
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0a0a0;
        font-weight: 300;
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.85rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #218838 0%, #1aa179 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    /* Progress bar styling */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #28a745;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #28a745;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.8rem;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Hide default header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def validate_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """Validate input parameters."""
    errors = []
    
    if not (0 <= N <= 200):
        errors.append("Nitrogen must be between 0 and 200")
    if not (0 <= P <= 200):
        errors.append("Phosphorus must be between 0 and 200")
    if not (0 <= K <= 200):
        errors.append("Potassium must be between 0 and 200")
    if not (-10 <= temperature <= 50):
        errors.append("Temperature must be between -10 and 50°C")
    if not (0 <= humidity <= 100):
        errors.append("Humidity must be between 0 and 100%")
    if not (0 <= ph <= 14):
        errors.append("pH must be between 0 and 14")
    if not (0 <= rainfall <= 500):
        errors.append("Rainfall must be between 0 and 500mm")
    
    return errors


def create_confidence_chart(top_crops):
    """Create a bar chart for crop confidence."""
    crops = [c[0].capitalize() for c in top_crops]
    confidences = [c[1] for c in top_crops]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#28a745' if i == 0 else '#17a2b8' if i == 1 else '#6c757d' 
              for i in range(len(crops))]
    
    bars = ax.barh(crops, confidences, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{conf:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, max(confidences) * 1.2)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('🌾 Top Crop Predictions', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_profit_chart(profit_comparison):
    """Create a bar chart for profit comparison."""
    if not profit_comparison:
        return None
    
    crops = [p['crop'].capitalize() for p in profit_comparison[:5]]
    profits = [p['net_profit'] for p in profit_comparison[:5]]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(crops)))
    
    bars = ax.barh(crops, [p/1000 for p in profits], color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, profit in zip(bars, profits):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'₹{profit:,}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, max(profits)/1000 * 1.2)
    ax.set_xlabel('Net Profit (₹ in thousands/hectare)', fontsize=12)
    ax.set_title('💰 Profit Comparison', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_feature_importance_chart():
    """Create a feature importance chart."""
    try:
        model_data = load_model()
        model = model_data['model']
        
        importance = model.feature_importances_
        df = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        
        bars = ax.barh(df['Feature'], df['Importance'], color=colors, edgecolor='white')
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('📊 Feature Importance', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    except:
        return None


# ============================================================================
# ============================================================================
# MAIN PAGE
# ============================================================================
def main():
    """Main function to run the premium Streamlit app."""
    
    # ========================================
    # 1. HERO SECTION
    # ========================================
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🌱 Smart Agriculture AI Advisor</div>
        <div class="hero-subtitle">AI-powered crop & profit recommendation system</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # 2. SIDEBAR - Input Parameters
    # ========================================
    st.sidebar.markdown("## 🌱 Input Parameters")
    st.sidebar.markdown("Adjust the sliders to match your soil and weather conditions.")
    
    # Soil Nutrients Section
    st.sidebar.markdown("### 🧪 Soil Nutrients")
    N = st.sidebar.slider("Nitrogen (N)", min_value=0, max_value=200, value=90, help="Nitrogen content in soil (kg/ha)")
    P = st.sidebar.slider("Phosphorus (P)", min_value=0, max_value=200, value=40, help="Phosphorus content in soil (kg/ha)")
    K = st.sidebar.slider("Potassium (K)", min_value=0, max_value=200, value=40, help="Potassium content in soil (kg/ha)")
    
    # Environmental Conditions Section
    st.sidebar.markdown("### 🌡️ Environmental Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", min_value=-10, max_value=50, value=20, help="Average temperature in Celsius")
    humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=80, help="Humidity percentage")
    ph = st.sidebar.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Soil pH value (0-14)")
    rainfall = st.sidebar.slider("Rainfall (mm)", min_value=0, max_value=500, value=200, help="Annual rainfall in mm")
    
    # ========================================
    # 3. SYSTEM HIGHLIGHTS
    # ========================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI Model**\n\nRandom Forest Classifier with 200 trees for accurate predictions")
    
    with col2:
        st.info("📊 **Smart Insights**\n\nTop 3 crop recommendations with confidence scores")
    
    with col3:
        st.info("💰 **Profit Engine**\n\nReal-time profit analysis based on market prices")
    
    st.markdown("---")
    
    # ========================================
    # 4. INPUT SUMMARY & MODEL STATUS
    # ========================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Your Input Summary")
        summary_data = {
            'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                        'Temperature', 'Humidity', 'Soil pH', 'Rainfall'],
            'Value': [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha",
                     f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall} mm"]
        }
        st.table(pd.DataFrame(summary_data))
    
    with col2:
        st.markdown("### ⚡ System Status")
        if not is_model_trained():
            st.warning("⚠️ Model not trained yet!")
            if st.button("🔧 Train Model Now"):
                with st.spinner("Training model... This may take a minute."):
                    try:
                        train_model()
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.success("✅ AI Model Ready")
    
    st.markdown("---")
    
    # ========================================
    # 5. PREDICTION BUTTON
    # ========================================
    st.markdown("### 🔮 Get Your Crop Recommendation")
    
    if st.button("🌾 Predict Best Crops", use_container_width=True):
        # Validate inputs
        errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        
        if errors:
            st.error("⚠️ Please fix the following errors:")
            for error in errors:
                st.write(f"- {error}")
        else:
            try:
                # Make prediction
                with st.spinner("🤔 AI is analyzing your conditions..."):
                    result = predict_crop(N, P, K, temperature, humidity, ph, rainfall, top_n=3)
                
                if result['success']:
                    # ========================================
                    # 6. KPI DASHBOARD
                    # ========================================
                    st.markdown("### 🎯 Prediction Results")
                    
                    best_crop = result['ai_prediction']['best_crop']
                    confidence = result['ai_prediction']['confidence']
                    top_crops = result['ai_prediction']['top_crops']
                    
                    # KPI Cards using st.metric
                    kpi1, kpi2, kpi3 = st.columns(3)
                    
                    with kpi1:
                        st.metric("🌾 Best Crop", best_crop.capitalize())
                    
                    with kpi2:
                        st.metric("📊 AI Confidence", f"{confidence:.1f}%")
                    
                    with kpi3:
                        profit_data = calculate_profit(best_crop)
                        if profit_data:
                            st.metric("💰 Est. Profit", f"₹{profit_data['net_profit']:,}/ha")
                    
                    st.markdown("---")
                    
                    # ========================================
                    # 7. TOP 3 CROPS WITH PROGRESS BARS
                    # ========================================
                    st.markdown("### 📈 Top 3 Crop Predictions")
                    
                    for i, (crop, conf) in enumerate(top_crops):
                        # Create progress bar
                        progress_html = f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: 600; font-size: 1.1rem;">{i+1}. {crop.capitalize()}</span>
                                <span style="font-weight: 700; color: #28a745;">{conf:.1f}%</span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                                <div style="width: {conf}%; background: linear-gradient(90deg, #28a745, #20c997); height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(progress_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # ========================================
                    # 8. ADVANCED VISUALIZATIONS
                    # ========================================
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.markdown("#### 🌾 Crop vs Confidence")
                        fig1 = create_confidence_chart(top_crops)
                        st.pyplot(fig1)
                    
                    with chart_col2:
                        st.markdown("#### 💰 Crop vs Profit")
                        profit_comparison = result['profit_analysis']['comparison']
                        fig2 = create_profit_chart(profit_comparison)
                        if fig2:
                            st.pyplot(fig2)
                    
                    st.markdown("---")
                    
                    # ========================================
                    # 9. PROFIT INTELLIGENCE
                    # ========================================
                    st.markdown("### 💰 Profit Intelligence")
                    
                    profit_col1, profit_col2 = st.columns(2)
                    
                    with profit_col1:
                        most_profitable = result['profit_analysis']['most_profitable']
                        most_profitable_profit = result['profit_analysis']['most_profitable_profit']
                        
                        st.markdown(f"""
                        **📈 Most Profitable Crop:** {most_profitable.capitalize()}
                        
                        **Estimated Profit:** ₹{most_profitable_profit:,}/hectare
                        """)
                        
                        if best_crop.lower() != most_profitable.lower():
                            st.warning(f"💡 Note: The most profitable crop ({most_profitable.capitalize()}) differs from the AI-recommended crop ({best_crop.capitalize()}). Consider both options!")
                    
                    with profit_col2:
                        if profit_data:
                            st.markdown(f"""
                            **📊 ROI for {best_crop.capitalize()}:** {profit_data['roi']:.1f}%
                            
                            **Cultivation Cost:** ₹{profit_data['cost_of_cultivation']:,}/hectare
                            
                            **Duration:** {profit_data['duration_months']} months
                            """)
                    
                    st.markdown("---")
                    
                    # ========================================
                    # 10. EXPLAINABLE AI
                    # ========================================
                    st.markdown("### 💡 Why This Crop?")
                    
                    # Generate custom explanation based on conditions
                    explanation_parts = []
                    
                    # Rainfall analysis
                    if rainfall > 200:
                        explanation_parts.append("✓ High rainfall (>200mm) supports water-intensive crops like rice and sugarcane")
                    elif rainfall < 100:
                        explanation_parts.append("✓ Low rainfall (<100mm) favors drought-tolerant crops like millets and groundnuts")
                    else:
                        explanation_parts.append("✓ Moderate rainfall suits a wide range of crops")
                    
                    # Temperature analysis
                    if temperature > 30:
                        explanation_parts.append("✓ High temperature favors warm-season crops like cotton and maize")
                    elif temperature < 20:
                        explanation_parts.append("✓ Cool temperature supports wheat, potato, and coffee")
                    else:
                        explanation_parts.append("✓ Moderate temperature is ideal for most crops")
                    
                    # Soil nutrients
                    if N > 70:
                        explanation_parts.append("✓ High nitrogen content supports leafy crops like rice and wheat")
                    if P > 50:
                        explanation_parts.append("✓ Good phosphorus benefits root and fruit development")
                    
                    # pH analysis
                    if ph > 7:
                        explanation_parts.append("✓ Alkaline soil suits cotton and sugarcane")
                    elif ph < 6:
                        explanation_parts.append("✓ Acidic soil favors tea, coffee, and rice")
                    else:
                        explanation_parts.append("✓ Neutral pH is optimal for most crops")
                    
                    # Display explanation
                    for part in explanation_parts:
                        st.markdown(f"- {part}")
                    
                    st.markdown(f"""
                    **🎯 AI Conclusion:** Based on your conditions (Temperature: {temperature}°C, Humidity: {humidity}%, pH: {ph}, Rainfall: {rainfall}mm), **{best_crop.capitalize()}** is the optimal choice with {confidence:.1f}% confidence.
                    """)
                    
                    # ========================================
                    # 11. FEATURE IMPACT ANALYSIS
                    # ========================================
                    with st.expander("🔍 View Feature Impact Analysis"):
                        try:
                            impact = get_feature_impact(N, P, K, temperature, humidity, ph, rainfall)
                            impact_df = pd.DataFrame(impact['feature_importance'])
                            impact_df['Importance'] = impact_df['importance'].apply(lambda x: f"{x:.3f}")
                            impact_df['Impact'] = impact_df['impact'].apply(lambda x: f"{x:.3f}")
                            
                            st.table(impact_df[['Feature', 'Importance', 'Value', 'Impact']].rename(
                                columns={'Feature': 'Feature', 'Importance': 'Importance Score', 
                                        'Value': 'Your Value', 'Impact': 'Impact Score'}
                            ))
                            
                            st.info(f"💡 Most impactful factor: **{impact['most_impactful']}**")
                        except Exception as e:
                            st.warning("Feature analysis暂时不可用")
                    
                else:
                    st.error("⚠️ Prediction failed. Please try again.")
                    
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                st.info("💡 Make sure the model is trained. Click 'Train Model Now' in the sidebar.")
    
    # ========================================
    # 12. DATA INSIGHTS SECTION
    # ========================================
    st.markdown("---")
    
    if st.checkbox("📊 Show Data Insights & Analysis"):
        try:
            df = load_crop_data()
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Smart Agriculture Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. Crop distribution
            ax1 = axes[0, 0]
            crop_counts = df['label'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
            ax1.bar(crop_counts.index, crop_counts.values, color=colors)
            ax1.set_xlabel('Crop Type')
            ax1.set_ylabel('Count')
            ax1.set_title('Crop Distribution in Dataset')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Temperature vs Humidity
            ax2 = axes[0, 1]
            for crop in df['label'].unique()[:6]:
                subset = df[df['label'] == crop]
                ax2.scatter(subset['temperature'], subset['humidity'], 
                           label=crop, alpha=0.6, s=30)
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Humidity (%)')
            ax2.set_title('Temperature vs Humidity by Crop')
            ax2.legend(loc='best', fontsize=8)
            
            # 3. NPK distribution
            ax3 = axes[1, 0]
            npk_means = df.groupby('label')[['N', 'P', 'K']].mean()
            npk_means.plot(kind='bar', ax=ax3, width=0.8)
            ax3.set_xlabel('Crop Type')
            ax3.set_ylabel('Average Value')
            ax3.set_title('Average NPK Values by Crop')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(['Nitrogen', 'Phosphorus', 'Potassium'])
            
            # 4. pH distribution
            ax4 = axes[1, 1]
            ax4.hist(df['ph'], bins=20, color='green', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('pH Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Soil pH Distribution')
            ax4.axvline(df['ph'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {df["ph"].mean():.2f}')
            ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance
            st.markdown("#### 📊 Model Feature Importance")
            fig_imp = create_feature_importance_chart()
            if fig_imp:
                st.pyplot(fig_imp)
            
            # Profit analysis table
            st.markdown("#### 💰 Crop Profitability Analysis")
            profit_df = get_all_profits()
            st.dataframe(profit_df[['crop', 'net_profit', 'roi', 'water_requirement', 'season']].rename(
                columns={'crop': 'Crop', 'net_profit': 'Net Profit (₹)', 
                        'roi': 'ROI (%)', 'water_requirement': 'Water Need', 
                        'season': 'Season'}
            ).sort_values('Net Profit (₹)', ascending=False), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # ========================================
    # 13. REAL-WORLD IMPACT SECTION
    # ========================================
    st.markdown("---")
    st.markdown("### 🌍 Real-World Impact")
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.success("✅ **Helps Farmers**\n\nChoose the correct crop based on soil & weather")
    
    with impact_col2:
        st.success("✅ **Reduces Risk**\n\nMinimizes crop failure through data-driven decisions")
    
    with impact_col3:
        st.success("✅ **Improves Profit**\n\nIdentifies most profitable crop for maximum returns")
    
    with impact_col4:
        st.success("✅ **Sustainable**\n\nPromotes efficient resource usage in agriculture")
    
    # ========================================
    # 14. FOOTER
    # ========================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌱 <strong>Smart Agriculture AI Advisor</strong> | Powered by Machine Learning</p>
        <p>Made with ❤️ for sustainable farming | Demo Ready 🚀</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Check if model exists, if not train it
    if not is_model_trained():
        print("Model not found. Training new model...")
        try:
            train_model()
            print("Model trained successfully!")
        except Exception as e:
            print(f"Error training model: {e}")
    
    # Run the Streamlit app
    main()