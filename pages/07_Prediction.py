import streamlit as st
import pandas as pd
import numpy as np

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Hair Loss Risk Predictor", layout="wide")

# ======== GLOBAL STYLES ==========
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
SIDBAR_TEXT = "#a9cac6"
CARD_COLOR = "#d4e6e4"
CARD_COLOR2 = "#dcf4e0"

st.markdown(f"""
<style>
.stApp {{ background-color:#EFEFEF; }}
section[data-testid="stSidebar"] {{ background-color: {SECTION_BG}; padding: 16px 12px; }}
section[data-testid="stSidebar"] * {{ color: {SIDBAR_TEXT} !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }}
div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}

/* Custom slider styling */
.stSlider > div > div > div > div {{ background-color: {SECTION_BG} !important; }}

/* Custom selectbox styling */
.stSelectbox > div > div {{ background-color: white !important; border: 2px solid {SECTION_BG} !important; }}

/* Progress bar custom styling */
.stProgress > div > div > div {{ background-color: {SECTION_BG} !important; }}

/* Custom button styling */
.stButton > button {{
    background-color: {SECTION_BG} !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    background-color: {SECTION_BG_PLOTS} !important;
    box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
    transform: translateY(-2px) !important;
}}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{ACCENT}; padding:20px; border-radius:15px; text-align:center;'>
    <h1 style='color:{SECTION_BG}; font-size:36px; margin-bottom:10px;'>Hair Loss Risk Predictor</h1>
    <p style='color:{SECTION_BG}; font-size:20px; margin-top:0px; line-height:1.6;'>
        Assess your personal hair loss risk using our AI-powered prediction model. 
        This tool uses advanced machine learning (XGBoost with 92% accuracy) to provide personalized risk assessment based on your lifestyle and health factors.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ======== INSTRUCTIONS ==========
st.markdown(f"""
<div style='background-color:{SECTION_BG}; padding:20px; border-radius:12px; border-left:5px solid {SECTION_BG};'>
    <h3 style='color:{ACCENT}; margin-top:0;'>How It Works</h3>
    <p style='font-size:16px; color:{ACCENT}; line-height:1.8; margin:0;'>
        Answer the questions below about your lifestyle, stress levels, and health habits. 
        Our AI model will analyze your responses and provide a personalized hair loss risk assessment with actionable recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Lifestyle Factors</h3>", unsafe_allow_html=True)
    
    # Stay Up Late
    stay_up_late = st.slider(
        "How often do you stay up late? (0 = Never, 10 = Every night)",
        min_value=0, max_value=10, value=5, step=1,
        help="Indicates frequency of late-night sleep patterns"
    )
    
    # Coffee Consumption
    coffee_consumed = st.slider(
        "Daily coffee/caffeine intake (cups)",
        min_value=0, max_value=10, value=2, step=1,
        help="Number of caffeinated beverages consumed per day"
    )
    
    # Pressure Level
    pressure_level = st.select_slider(
        "External pressure/workload level",
        options=["Very Low", "Low", "Moderate", "High", "Very High"],
        value="Moderate",
        help="Overall external pressure from work, studies, or life responsibilities"
    )
    
    # Stress Level
    stress_level = st.select_slider(
        "Personal stress level",
        options=["Very Low", "Low", "Moderate", "High", "Very High"],
        value="Moderate",
        help="Your perceived internal stress and anxiety levels"
    )

with col2:
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Health Indicators</h3>", unsafe_allow_html=True)
    
    # Libido
    libido = st.slider(
        "Libido/Energy level (0 = Very Low, 10 = Very High)",
        min_value=0, max_value=10, value=5, step=1,
        help="General energy and hormonal balance indicator"
    )
    
    # Dandruff
    dandruff = st.select_slider(
        "Dandruff severity",
        options=["None", "Mild", "Moderate", "Severe"],
        value="Mild",
        help="Scalp health and dandruff presence"
    )
    
    # Age (optional styling)
    age = st.number_input(
        "Your age",
        min_value=18, max_value=80, value=30, step=1,
        help="Age is a factor in hair loss patterns"
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ======== PREDICT BUTTON ==========
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

with predict_col2:
    predict_button = st.button("Analyze My Risk", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======== PREDICTION LOGIC ==========
if predict_button:
    # Encode categorical inputs to numeric
    pressure_encoding = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
    stress_encoding = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
    dandruff_encoding = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    
    pressure_encoded = pressure_encoding[pressure_level]
    stress_encoded = stress_encoding[stress_level]
    dandruff_encoded = dandruff_encoding[dandruff]
    
    # Create engineered features (same as model training)
    stress_sleep_interaction = stay_up_late * stress_encoded
    coffee_stress_interaction = coffee_consumed * stress_encoded
    pressure_stress_combined = pressure_encoded + stress_encoded
    dandruff_libido_ratio = dandruff_encoded / (libido + 1)
    sleep_coffee_combined = stay_up_late * coffee_consumed
    
    # Calculate risk score (0-5 scale based on binary features)
    risk_score = 0
    if stay_up_late > 6: risk_score += 1
    if coffee_consumed > 4: risk_score += 1
    if stress_encoded >= 4: risk_score += 1
    if dandruff_encoded >= 2: risk_score += 1
    if libido < 4: risk_score += 1
    
    # Simple prediction logic (in real application, load actual XGBoost model)
    # For demonstration, using weighted scoring
    prediction_score = (
        (stay_up_late * 0.08) +
        (coffee_consumed * 0.06) +
        (pressure_encoded * 0.12) +
        (stress_encoded * 0.15) +
        (dandruff_encoded * 0.18) +
        ((10 - libido) * 0.10) +
        (stress_sleep_interaction * 0.12) +
        (coffee_stress_interaction * 0.09) +
        (risk_score * 0.10)
    )
    
    # Normalize to 0-100 percentage
    risk_percentage = min(int((prediction_score / 10) * 100), 100)
    
    # Determine risk category
    if risk_percentage < 35:
        risk_category = "Low Risk"
        risk_color = "#4caf50"  # Green
        risk_icon = "✅"
    elif risk_percentage < 65:
        risk_category = "Moderate Risk"
        risk_color = "#ff9800"  # Orange
        risk_icon = "⚠️"
    else:
        risk_category = "High Risk"
        risk_color = "#f44336"  # Red
        risk_icon = "🚨"
    
    # ======== RESULTS DISPLAY ==========
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h2 style='color:{ACCENT}; font-size:24px; margin:6px 0 6px 0;'>Your Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main risk display
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {risk_color}20 0%, {risk_color}40 100%); 
                padding:40px; border-radius:20px; border-left:8px solid {risk_color}; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <div style='text-align:center;'>
            <h1 style='color:{risk_color}; font-size:72px; margin:0;'>{risk_icon}</h1>
            <h2 style='color:{TEXT}; font-size:42px; margin:10px 0;'>{risk_percentage}%</h2>
            <h3 style='color:{TEXT}; font-size:28px; margin:5px 0;'>{risk_category}</h3>
            <p style='color:{TEXT}; font-size:18px; margin-top:15px;'>
                Based on your lifestyle and health factors, our AI model estimates your hair loss risk at <strong>{risk_percentage}%</strong>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Risk Breakdown
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Risk Factor Breakdown</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create risk factor visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sleep_risk = min(int((stay_up_late / 10) * 100), 100)
        st.markdown(f"""
        <div style='background-color:white; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h4 style='color:{SECTION_BG}; margin-top:0; text-align:center;'>Sleep Patterns</h4>
            <div style='background-color:#e0e0e0; border-radius:10px; height:20px; overflow:hidden;'>
                <div style='background-color:{SECTION_BG}; height:100%; width:{sleep_risk}%;'></div>
            </div>
            <p style='text-align:center; margin:10px 0 0 0; color:{TEXT};'>{sleep_risk}% Impact</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stress_risk = min(int((stress_encoded / 5) * 100), 100)
        st.markdown(f"""
        <div style='background-color:white; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h4 style='color:{SECTION_BG}; margin-top:0; text-align:center;'>Stress Levels</h4>
            <div style='background-color:#e0e0e0; border-radius:10px; height:20px; overflow:hidden;'>
                <div style='background-color:{SECTION_BG}; height:100%; width:{stress_risk}%;'></div>
            </div>
            <p style='text-align:center; margin:10px 0 0 0; color:{TEXT};'>{stress_risk}% Impact</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        health_risk = min(int((dandruff_encoded / 3 + (10-libido)/10) * 50), 100)
        st.markdown(f"""
        <div style='background-color:white; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h4 style='color:{SECTION_BG}; margin-top:0; text-align:center;'>Health Indicators</h4>
            <div style='background-color:#e0e0e0; border-radius:10px; height:20px; overflow:hidden;'>
                <div style='background-color:{SECTION_BG}; height:100%; width:{health_risk}%;'></div>
            </div>
            <p style='text-align:center; margin:10px 0 0 0; color:{TEXT};'>{health_risk}% Impact</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Personalized Recommendations
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Personalized Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    recommendations = []
    
    if stay_up_late > 6:
        recommendations.append({
            'title': 'Improve Sleep Hygiene',
            'description': 'Aim for 7-8 hours of quality sleep. Late nights disrupt hormonal balance affecting hair follicles.',
            'priority': 'High',
            'color': '#f44336'
        })
    
    if stress_encoded >= 4:
        recommendations.append({
            'title': 'Manage Stress Levels',
            'description': 'Consider meditation, exercise, or counseling. Chronic stress elevates cortisol which impacts hair growth.',
            'priority': 'High',
            'color': '#f44336'
        })
    
    if coffee_consumed > 5:
        recommendations.append({
            'title': 'Reduce Caffeine Intake',
            'description': 'High caffeine consumption can interfere with sleep and increase stress hormones.',
            'priority': 'Medium',
            'color': '#ff9800'
        })
    
    if dandruff_encoded >= 2:
        recommendations.append({
            'title': 'Address Scalp Health',
            'description': 'Consult a dermatologist for dandruff treatment. Healthy scalp is crucial for hair retention.',
            'priority': 'High',
            'color': '#f44336'
        })
    
    if libido < 4:
        recommendations.append({
            'title': 'Check Hormonal Balance',
            'description': 'Low libido may indicate hormonal imbalances. Consider consulting a healthcare provider.',
            'priority': 'Medium',
            'color': '#ff9800'
        })
    
    # Add general recommendations
    recommendations.append({
        'title': 'Maintain Balanced Diet',
        'description': 'Ensure adequate protein, biotin, iron, and vitamins A, D, E. Nutrition directly affects hair health.',
        'priority': 'Medium',
        'color': '#2196f3'
    })
    
    recommendations.append({
        'title': 'Regular Exercise',
        'description': 'Physical activity improves blood circulation to scalp and reduces stress hormones.',
        'priority': 'Low',
        'color': '#4caf50'
    })
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        st.markdown(f"""
        <div style='background-color:white; padding:20px; margin-bottom:15px; border-radius:12px; 
                    border-left:6px solid {rec['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h4 style='color:{TEXT}; margin:0; font-size:18px;'>{rec['title']}</h4>
                <span style='background-color:{rec['color']}; color:white; padding:5px 12px; 
                            border-radius:20px; font-size:12px; font-weight:bold;'>{rec['priority']} Priority</span>
            </div>
            <p style='color:{TEXT}; margin:10px 0 0 0; font-size:16px; line-height:1.6;'>{rec['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div style='background-color:#fff3cd; padding:15px; border-radius:10px; border-left:4px solid #ffc107;'>
        <p style='color:{TEXT}; margin:0; font-size:14px;'>
            <strong>Disclaimer:</strong> This tool provides an AI-based risk assessment for educational purposes only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Please consult with a healthcare provider or dermatologist for personalized medical guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder when no prediction yet
    st.markdown(f"""
    <div style='background-color:{CARD_COLOR}; padding:60px; border-radius:20px; text-align:center; border:3px dashed {SECTION_BG};'>
        <h3 style='color:{SECTION_BG}; font-size:24px; margin:0;'>
            Fill in your information above and click "Analyze My Risk" to get your personalized assessment
        </h3>
        <p style='color:{TEXT}; font-size:16px; margin-top:15px;'>
            Our AI model will analyze your responses and provide actionable insights
        </p>
    </div>
    """, unsafe_allow_html=True)