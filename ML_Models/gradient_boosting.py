import streamlit as st
import pandas as pd

# Color scheme
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
ACCENT = "#FFFFFF"
TEXT = "#2C3E50"
CARD_COLOR = "#d4e6e4"

def render_gradient_boosting_page():
    """Render Gradient Boosting model page"""
    
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:15px; text-align:center; border-radius:12px;'>
        <h2 style='color:{ACCENT}; font-size:28px; margin:5px 0;'>Model 4: Gradient Boosting with SMOTE (3 Classes)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Description
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class='feature-box'>
            <h3 style='color:{SECTION_BG}; margin-top:0;'>📋 Model Overview</h3>
            <p><strong>Algorithm:</strong> Gradient Boosting (Sklearn)</p>
            <p><strong>Target Classes:</strong> 3 (Low, Medium, Severe)</p>
            <p><strong>Number of Estimators:</strong> 300</p>
            <p><strong>Learning Rate:</strong> 0.05</p>
            <p><strong>Max Depth:</strong> 5</p>
            <p><strong>Subsample:</strong> 0.8 (Stochastic GB)</p>
            <p><strong>Resampling:</strong> SMOTE in pipeline</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='feature-box'>
            <h3 style='color:{SECTION_BG}; margin-top:0;'>🔧 Features & Approach</h3>
            <p><strong>Same 11 Features</strong> as other models</p>
            <p><strong>Gradient Boosting Characteristics:</strong></p>
            <ul>
                <li><strong>Sequential Learning:</strong> Each tree corrects previous errors</li>
                <li><strong>Gradient Descent:</strong> Optimizes loss function directly</li>
                <li><strong>Stochastic Sampling:</strong> 80% data per tree for robustness</li>
                <li><strong>More Interpretable:</strong> Easier to understand than XGBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Model Evaluation
    st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:20px;'>📊 Model Evaluation</h3>", unsafe_allow_html=True)

    evaluation_data_gb = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
                   'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
        'Value': ['0.9000', '0.8980 ± 0.0380', '0.8970', '0.9000', '0.8985', 'Medium (~20 sec)']
    })

    st.dataframe(evaluation_data_gb, use_container_width=True, hide_index=True)

    # Per-Class Performance
    st.markdown(f"<h4 style='color:{SECTION_BG};'>Per-Class Performance</h4>", unsafe_allow_html=True)
    class_perf_gb = pd.DataFrame({
        'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
        'Precision': [0.92, 0.87, 0.90],
        'Recall': [0.94, 0.85, 0.91],
        'F1-Score': [0.93, 0.86, 0.90]
    })
    st.dataframe(class_perf_gb, use_container_width=True, hide_index=True)

    # Problems and Solutions
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class='problem-box'>
            <h4 style='color:#cc0000; margin-top:0;'>⚠️ Shortcomings/Problems</h4>
            <ul>
                <li><strong>Sequential Training:</strong> Cannot parallelize like Random Forest, slower training</li>
                <li><strong>Sensitive to Outliers:</strong> Gradient descent can be affected by extreme values</li>
                <li><strong>Learning Rate Tuning:</strong> Requires careful balance between speed and accuracy</li>
                <li><strong>Memory Intensive:</strong> Stores all trees in memory</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='solution-box'>
            <h4 style='color:#006600; margin-top:0;'>✅ How We Fixed It</h4>
            <ul>
                <li><strong>Stochastic Sampling:</strong> Used subsample=0.8 to make training faster and more robust</li>
                <li><strong>Conservative Learning Rate:</strong> Set to 0.05 with 300 estimators for stable convergence</li>
                <li><strong>Shallow Trees:</strong> Max_depth=5 prevents overfitting and reduces memory</li>
                <li><strong>SMOTE Integration:</strong> Applied within pipeline to handle class imbalance properly</li>
                <li><strong>Feature Scaling:</strong> Standardized numeric features to reduce outlier impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Takeaways
    st.markdown(f"""
    <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:12px; border-left:5px solid {SECTION_BG};'>
        <h4 style='color:{SECTION_BG}; margin-top:0;'>💡 Key Takeaways</h4>
        <ul style='font-size:18px; line-height:1.8;'>
            <li><strong>Strong Performance:</strong> 90% accuracy - competitive with other ensemble methods</li>
            <li><strong>Good Alternative:</strong> Provides similar results to XGBoost with more interpretability</li>
            <li><strong>Reliable Predictions:</strong> Consistent performance across cross-validation folds</li>
            <li><strong>Sequential Strength:</strong> Each tree specifically corrects previous errors</li>
            <li><strong>Practical Choice:</strong> Good balance of accuracy and explainability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)