import streamlit as st
import pandas as pd

# Color scheme
BASE_BG = "#FFFFFF"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
ACCENT = "#FFFFFF"
TEXT = "#2C3E50"
CARD_COLOR = "#d4e6e4"

def render_gradient_boosting_page():
    """Render Gradient Boosting model page"""
    
    # Main Model Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>
        <h2 style='color:{ACCENT}; font-size:24px; margin:6px 0 6px 0;'>Model 4: Gradient Boosting with SMOTE (3 Classes)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Overview - Simple list format
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Model Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
        <li><b>Algorithm:</b> Gradient Boosting (Sklearn)</li>
        <li><b>Target Classes:</b> 3 (Low, Medium, Severe)</li>
        <li><b>Number of Estimators:</b> 300</li>
        <li><b>Learning Rate:</b> 0.05</li>
        <li><b>Max Depth:</b> 5</li>
        <li><b>Subsample:</b> 0.8 (Stochastic GB)</li>
        <li><b>Resampling:</b> SMOTE in pipeline</li>
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
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Sequential Error Correction</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Each tree in the sequence specifically targets and corrects the residual errors from previous trees, 
                using gradient descent optimization to minimize loss function and improve predictions iteratively.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:#A8D5BA; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Stochastic Sampling</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Randomly samples 80% of training data for each tree (subsample=0.8), introducing controlled variance 
                that prevents overfitting and makes the model more robust to outliers and noise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div style='background-color:#FFB347; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Shallow Tree Architecture</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Uses max_depth=5 to create simpler trees that capture main patterns without memorizing noise, 
                resulting in better generalization and reduced memory footprint compared to deeper alternatives.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background-color:#D9D9D9; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:{TEXT}; margin-top:0; font-size:18px;'>Gradient-Based Learning</h4>
            <p style='color:{TEXT}; font-size:15px; line-height:1.6; margin:0;'>
                Optimizes loss function directly using gradient descent with learning_rate=0.05, 
                allowing fine-tuned control over convergence speed and preventing overshooting optimal solutions.
            </p>
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

    evaluation_data_gb = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
                   'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
        'Value': ['0.9000', '0.8980 ± 0.0380', '0.8970', '0.9000', '0.8985', 'Medium (~20 sec)']
    })

    st.dataframe(evaluation_data_gb, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-Class Performance
    st.markdown(f"<h4 style='color:{SECTION_BG}; font-size:21px;'>Per-Class Performance</h4>", unsafe_allow_html=True)
    class_perf_gb = pd.DataFrame({
        'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
        'Precision': [0.92, 0.87, 0.90],
        'Recall': [0.94, 0.85, 0.91],
        'F1-Score': [0.93, 0.86, 0.90],
        'Support': [45, 52, 53]
    })
    st.dataframe(class_perf_gb, use_container_width=True, hide_index=True)

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
                <li><b>Sequential Training:</b> Cannot parallelize like Random Forest, resulting in slower training times</li>
                <li><b>Sensitive to Outliers:</b> Gradient descent can be disproportionately affected by extreme values in data</li>
                <li><b>Learning Rate Tuning:</b> Requires careful balance between convergence speed and model accuracy</li>
                <li><b>Memory Intensive:</b> Stores all 300 trees in memory which can be resource-demanding</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#e6ffe6; padding:15px; border-radius:8px; border-left:4px solid #44ff44;'>
            <h4 style='color:#006600; margin-top:0; font-size:18px;'>How I Fixed It</h4>
            <ul style='font-size:16px;'>
                <li><b>Stochastic Sampling:</b> Used subsample=0.8 to make training faster and more robust to outliers</li>
                <li><b>Conservative Learning Rate:</b> Set to 0.05 with 300 estimators for stable convergence without overshooting</li>
                <li><b>Shallow Trees:</b> Max_depth=5 prevents overfitting, reduces memory usage, and improves generalization</li>
                <li><b>SMOTE Integration:</b> Applied within pipeline to handle class imbalance while preventing data leakage</li>
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
            <li><b>Strong Performance:</b> 90% accuracy demonstrates competitive results with other ensemble methods</li>
            <li><b>Good Alternative:</b> Provides similar results to XGBoost with more interpretability and simpler implementation</li>
            <li><b>Reliable Predictions:</b> Consistent performance across cross-validation folds (CV std=0.038)</li>
            <li><b>Sequential Strength:</b> Each tree specifically targets and corrects errors from previous predictions</li>
            <li><b>Practical Choice:</b> Excellent balance of accuracy, explainability, and ease of deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)