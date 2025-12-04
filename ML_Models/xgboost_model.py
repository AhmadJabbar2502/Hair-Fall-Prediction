import streamlit as st
import pandas as pd

# Color scheme
BASE_BG = "#FFFFFF"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
ACCENT = "#FFFFFF"
TEXT = "#2C3E50"
CARD_COLOR = "#d4e6e4"

def render_xgboost_page():
    """Render XGBoost model page"""
    
    # Main Model Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>
        <h2 style='color:{ACCENT}; font-size:24px; margin:6px 0 6px 0;'>Model 3: XGBoost with SMOTE (3 Classes)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Overview - Simple list format
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Model Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
        <li><b>Algorithm:</b> XGBoost (Extreme Gradient Boosting)</li>
        <li><b>Target Classes:</b> 3 (mapped to 0, 1, 2 for XGBoost)</li>
        <li><b>Number of Estimators:</b> 300</li>
        <li><b>Learning Rate:</b> 0.05</li>
        <li><b>Max Depth:</b> 6</li>
        <li><b>Subsample:</b> 0.8</li>
        <li><b>Colsample by Tree:</b> 0.8</li>
        <li><b>Objective:</b> multi:softprob</li>
        <li><b>Eval Metric:</b> mlogloss</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Base Features - Bullet points
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Base Features (6)</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
        <li>Stay_Up_Late</li>
        <li>Coffee_Consumed</li>
        <li>Libido</li>
        <li>Pressure_Level_Encoding</li>
        <li>Stress_Level_Encoding</li>
        <li>Dandruff_Encoding</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Engineered Features Section
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Engineered Features (5)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 1
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>stress_sleep_interaction</b><br>
            Captures the compound effect of staying up late combined with high stress levels. 
            The interaction term (Stay_Up_Late × Stress_Level) represents how sleep deprivation amplifies stress-related hair loss.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 2
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>coffee_stress_interaction</b><br>
            Represents the combined effect of caffeine consumption and stress levels. 
            High coffee intake during stressful periods may indicate heightened cortisol response, calculated as (Coffee_Consumed × Stress_Level).
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 3
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>pressure_stress_combined</b><br>
            Combines external pressure and internal stress into a single metric. 
            This additive feature (Pressure_Level + Stress_Level) captures overall psychological burden affecting hair health.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 4
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>dandruff_libido_ratio</b><br>
            Serves as a proxy for hormonal balance by comparing dandruff severity to libido levels. 
            The ratio (Dandruff / (Libido + 1)) may indicate DHT-related hormonal imbalances linked to hair loss.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 5
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>sleep_coffee_combined</b><br>
            Multiplicative interaction between late nights and caffeine consumption. 
            The product (Stay_Up_Late × Coffee_Consumed) captures the synergistic effect of poor sleep habits and stimulant use on hair follicle health.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Key Innovations Section
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Key Innovations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Innovation cards in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:#85ada6; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Second-Order Optimization</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Uses both gradient and Hessian (second derivative) information for more accurate optimization, 
                providing faster convergence and better handling of complex loss landscapes compared to first-order methods.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:#A8D5BA; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Built-in Regularization</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Incorporates L1 and L2 regularization penalties directly into the objective function, 
                preventing overfitting through automatic feature selection and weight shrinkage without manual tuning.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div style='background-color:#FFB347; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Parallel Tree Construction</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Leverages parallel processing to build trees efficiently using all CPU cores, 
                reducing training time by 3x compared to sequential gradient boosting while maintaining model quality.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background-color:#D9D9D9; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Stochastic Gradient Boosting</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Randomly samples 80% of data (subsample=0.8) and features (colsample_bytree=0.8) per tree, 
                introducing controlled randomness that improves generalization and reduces overfitting risk.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Hyperparameter Configuration
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Hyperparameter Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:15px; border-radius:8px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0; font-size:18px;'>Tree Parameters</h4>
            <ul style='font-size:16px;'>
                <li><b>n_estimators=300:</b> Number of boosting rounds (trees)</li>
                <li><b>max_depth=6:</b> Maximum tree depth (controls complexity)</li>
                <li><b>min_child_weight=1:</b> Minimum sum of instance weight in a child</li>
                <li><b>gamma=0:</b> Minimum loss reduction for split</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:15px; border-radius:8px;'>
            <h4 style='color:{SECTION_BG}; margin-top:0; font-size:18px;'>Regularization Parameters</h4>
            <ul style='font-size:16px;'>
                <li><b>learning_rate=0.05:</b> Step size shrinkage (eta)</li>
                <li><b>subsample=0.8:</b> Fraction of samples per tree</li>
                <li><b>colsample_bytree=0.8:</b> Fraction of features per tree</li>
                <li><b>objective=multi:softprob:</b> Multi-class probabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Model Evaluation Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Model Evaluation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    evaluation_data_xgb = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
                   'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'ROC-AUC Score', 'Training Time'],
        'Value': ['0.9200', '0.9180 ± 0.0320', '0.9180', '0.9200', '0.9190', '0.9650', 'Medium (~18 sec)']
    })

    st.dataframe(evaluation_data_xgb, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-Class Performance
    st.markdown(f"<h4 style='color:{SECTION_BG}; font-size:21px;'>Per-Class Performance</h4>", unsafe_allow_html=True)
    class_perf_xgb = pd.DataFrame({
        'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
        'Precision': [0.94, 0.90, 0.92],
        'Recall': [0.96, 0.88, 0.93],
        'F1-Score': [0.95, 0.89, 0.92],
        'Support': [45, 52, 53]
    })
    st.dataframe(class_perf_xgb, use_container_width=True, hide_index=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Why XGBoost Achieved Best Performance
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Why XGBoost Achieved Best Performance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color:#e8f5e9; padding:20px; border-radius:12px; border-left:5px solid #4caf50;'>
        <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
            <li><b>vs Logistic Regression:</b> Captures non-linear relationships through tree splits and handles feature interactions automatically</li>
            <li><b>vs Random Forest:</b> Sequential learning corrects previous errors; uses second-order gradients for more accurate optimization</li>
            <li><b>vs Gradient Boosting:</b> Parallel tree construction (3x faster) and built-in regularization prevent overfitting better</li>
            <li><b>Result:</b> XGBoost achieved 92% accuracy by optimally balancing bias-variance tradeoff through sophisticated regularization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Problems and Solutions Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Challenges and Solutions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Problems
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div style='background-color:#ffe6e6; padding:15px; border-radius:8px; border-left:4px solid #ff4444;'>
            <h4 style='color:#cc0000; margin-top:0; font-size:18px;'>Shortcomings/Problems</h4>
            <ul style='font-size:16px;'>
                <li><b>Label Encoding Required:</b> XGBoost needs 0-indexed labels; original classes (1,2,3) needed mapping to (0,1,2)</li>
                <li><b>Hyperparameter Sensitivity:</b> Performance varies significantly with learning rate and depth settings</li>
                <li><b>Black Box Nature:</b> Sequential tree building makes it less interpretable than logistic regression</li>
                <li><b>Overfitting Risk:</b> Deep trees without regularization caused 15% validation accuracy drop</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#e6ffe6; padding:15px; border-radius:8px; border-left:4px solid #44ff44;'>
            <h4 style='color:#006600; margin-top:0; font-size:18px;'>How I Fixed It</h4>
            <ul style='font-size:16px;'>
                <li><b>Automatic Label Mapping:</b> Implemented bidirectional mapping (1→0, 2→1, 3→2 for training; reverse for predictions)</li>
                <li><b>Conservative Learning Rate:</b> Set to 0.05 with 300 estimators for gradual learning and convergence</li>
                <li><b>Multi-Level Regularization:</b> Combined subsample=0.8, colsample_bytree=0.8, and max_depth=6</li>
                <li><b>Feature Importance:</b> Extracted gain-based feature importance for interpretability insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Key Takeaways Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Key Takeaways</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Takeaways in single box
    st.markdown(f"""
    <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:12px; border-left:5px solid {SECTION_BG};'>
        <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
            <li><b>Best Overall Accuracy:</b> 92% test accuracy - highest among all four models with ROC-AUC of 96.5%</li>
            <li><b>State-of-the-Art Algorithm:</b> Industry-standard approach used in 70%+ of Kaggle competition winning solutions</li>
            <li><b>Production Ready:</b> Recommended for deployment due to superior accuracy and reliability (CV std=0.032)</li>
            <li><b>Advanced Optimization:</b> Second-order gradients provide superior convergence compared to first-order methods</li>
            <li><b>Consistent Excellence:</b> Strong performance across all classes with no significant weak points</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)