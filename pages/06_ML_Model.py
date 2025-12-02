

import streamlit as st
import pandas as pd
from ML_Models.logistic_regression import render_logistic_page
from ML_Models.random_forest import render_random_forest_page
from ML_Models.xgboost_model import render_xgboost_page
from ML_Models.gradient_boosting import render_gradient_boosting_page

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Model Development and Evaluation", layout="wide")

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
div[data-testid="stMetricLabel"] {{ font-size: 22px !important; color: #333 !important; }}
div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}
.model-card {{
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 2px solid {SECTION_BG_PLOTS};
    margin-bottom: 20px;
}}
.feature-box {{
    background-color: {CARD_COLOR};
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}}
.problem-box {{
    background-color: #ffe6e6;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #ff4444;
}}
.solution-box {{
    background-color: #e6ffe6;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #44ff44;
}}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{SECTION_BG}; padding:20px; border-radius:15px; text-align:center;'>
    <h1 style='color:{ACCENT}; font-size:36px; margin-bottom:10px;'>Model Development and Evaluation</h1>
    <p style='color:{ACCENT}; font-size:20px; margin-top:0px; line-height:1.6;'>
        This section presents a comprehensive analysis of four distinct machine learning models developed to predict hair loss severity. 
        Each model is evaluated using multiple metrics, cross-validation techniques, and visualization tools to ensure robust performance 
        and demonstrate understanding of advanced model selection and validation strategies.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

# ======== MODEL SELECTOR ==========
model_option = st.selectbox(
    "Select Model:",
    ["None", "Model 1: Logistic Regression", "Model 2: Random Forest", 
     "Model 3: XGBoost", "Model 4: Gradient Boosting", "Model Comparison"],
    index=0
)

if model_option == "None":
    st.info("Select a model from the dropdown above to view detailed analysis.")
    
elif model_option == "Model 1: Logistic Regression":
    render_logistic_page()
    
elif model_option == "Model 2: Random Forest":
    render_random_forest_page()
    
elif model_option == "Model 3: XGBoost":
    render_xgboost_page()
    
elif model_option == "Model 4: Gradient Boosting":
    render_gradient_boosting_page()
    
elif model_option == "Model Comparison":
    # ============================================================================
    # MODEL COMPARISON
    # ============================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:20px; text-align:center; border-radius:15px;'>
        <h2 style='color:{ACCENT}; font-size:30px; margin:5px 0;'>📊 Comprehensive Model Comparison</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Table
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest + SMOTE', 'XGBoost + SMOTE', 'Gradient Boosting + SMOTE'],
        'Classes': [4, 3, 3, 3],
        'Test Accuracy': ['70.0%', '91.0%', '92.0%', '90.0%'],
        'CV Accuracy': ['70.0%', '91.0%', '91.8%', '89.8%'],
        'Training Time': ['~2 sec', '~15 sec', '~18 sec', '~20 sec'],
        'Interpretability': ['High', 'Medium', 'Low', 'Medium'],
        'Best For': ['Baseline', 'Balanced Performance', 'Highest Accuracy', 'Alternative Ensemble']
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Key Findings
    st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:30px;'>🔍 Key Findings</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>92%</h2>
            <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Best Accuracy</strong></p>
            <p style='color:{TEXT}; font-size:16px;'>XGBoost Model</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>21%</h2>
            <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Improvement</strong></p>
            <p style='color:{TEXT}; font-size:16px;'>From Baseline to Best Model</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>4</h2>
            <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Models Tested</strong></p>
            <p style='color:{TEXT}; font-size:16px;'>Comprehensive Evaluation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Final Recommendations
    st.markdown(f"""
    <div style='background-color:{CARD_COLOR2}; padding:20px; border-radius:12px; border-left:5px solid {SECTION_BG};'>
        <h3 style='color:{SECTION_BG}; margin-top:0;'>🎯 Final Recommendations</h3>
        <ul style='font-size:18px; line-height:1.8;'>
            <li><strong>Best Overall Model:</strong> XGBoost with SMOTE (92% accuracy, excellent per-class performance)</li>
            <li><strong>Most Balanced:</strong> Random Forest with SMOTE (91% accuracy, good interpretability)</li>
            <li><strong>Fastest Training:</strong> Logistic Regression (2 seconds, good for quick iterations)</li>
            <li><strong>Production Deployment:</strong> Recommend XGBoost for highest accuracy and robust predictions</li>
            <li><strong>Class Merging Impact:</strong> Merging classes 3 & 4 improved accuracy by 21 percentage points</li>
            <li><strong>Feature Engineering:</strong> Interaction features contributed 30-40% of model importance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validation Techniques Used
    st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:30px;'>✅ Validation Techniques Demonstrated</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0;'>Cross-Validation</h4>
            <ul style='font-size:16px;'>
                <li>5-fold Stratified K-Fold</li>
                <li>SMOTE applied within each fold</li>
                <li>Prevents data leakage</li>
                <li>Reports mean ± std deviation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0;'>Imbalance Handling</h4>
            <ul style='font-size:16px;'>
                <li>SMOTE (Synthetic Oversampling)</li>
                <li>Class weighting (balanced)</li>
                <li>Stratified sampling</li>
                <li>Class merging strategy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0;'>Evaluation Metrics</h4>
            <ul style='font-size:16px;'>
                <li>Accuracy, Precision, Recall, F1-Score</li>
                <li>Per-class performance analysis</li>
                <li>Confusion matrices</li>
                <li>ROC-AUC scores (where applicable)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0;'>Advanced Techniques</h4>
            <ul style='font-size:16px;'>
                <li>Feature engineering (5 new features)</li>
                <li>Ensemble methods (RF, XGB, GB)</li>
                <li>Hyperparameter tuning</li>
                <li>Pipeline architecture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)