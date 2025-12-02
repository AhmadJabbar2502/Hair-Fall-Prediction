import streamlit as st
import pandas as pd

# Color scheme
BASE_BG = "#FFFFFF"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
ACCENT = "#FFFFFF"
TEXT = "#2C3E50"
CARD_COLOR = "#d4e6e4"

def render_logistic_page():
    """Render Logistic Regression model page"""
    
    # Main Model Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>
        <h2 style='color:{ACCENT}; font-size:24px; margin:6px 0 6px 0;'>Model 1: Multinomial Logistic Regression (4 Classes)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Overview - Simple list format
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Model Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
        <li><b>Algorithm:</b> Multinomial Logistic Regression</li>
        <li><b>Target Classes:</b> 4 (Low, Medium, High, Severe)</li>
        <li><b>Solver:</b> LBFGS (Limited-memory BFGS)</li>
        <li><b>Max Iterations:</b> 2000</li>
        <li><b>Class Weighting:</b> Balanced</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Base Features - Simple list format
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Base Features (6)</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:16px; color:{TEXT};'>
        Stay_Up_Late, Coffee_Consumed, Libido, Pressure_Level_Encoding, Stress_Level_Encoding, Dandruff_Encoding
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Engineered Features Section - with SECTION_BG_PLOTS background
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

    # Model Evaluation Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px;'>
        <h3 style='color:{ACCENT}; font-size:21px; margin:6px 0 6px 0;'>Model Evaluation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    evaluation_data_log = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
                   'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
        'Value': ['0.7000', '0.7000 ± 0.0400', '0.6950', '0.6980', '0.6965', 'Fast (~2 sec)']
    })

    st.dataframe(evaluation_data_log, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-Class Performance - Increased font size
    st.markdown(f"<h4 style='color:{SECTION_BG}; font-size:19px;'>Per-Class Performance</h4>", unsafe_allow_html=True)
    class_perf_log = pd.DataFrame({
        'Class': ['Low (1)', 'Medium (2)', 'High (3)', 'Severe (4)'],
        'Precision': [0.72, 0.68, 0.67, 0.70],
        'Recall': [0.75, 0.65, 0.69, 0.72],
        'F1-Score': [0.73, 0.66, 0.68, 0.71]
    })
    st.dataframe(class_perf_log, use_container_width=True, hide_index=True)

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
                <li><b>Linear Assumption:</b> Cannot capture complex non-linear relationships between features</li>
                <li><b>4-Class Complexity:</b> Struggles to distinguish between High (3) and Severe (4) classes</li>
                <li><b>Class Imbalance:</b> Some classes have fewer samples, affecting performance</li>
                <li><b>Limited Feature Interactions:</b> Only learns linear combinations of features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#e6ffe6; padding:15px; border-radius:8px; border-left:4px solid #44ff44;'>
            <h4 style='color:#006600; margin-top:0; font-size:18px;'>How I Fixed It</h4>
            <ul style='font-size:16px;'>
                <li><b>Feature Engineering:</b> Created 5 interaction features to capture non-linear patterns</li>
                <li><b>Class Weighting:</b> Applied balanced weights to handle class imbalance</li>
                <li><b>Merged Classes:</b> Combined classes 3 & 4 in subsequent models (improved to 91% accuracy)</li>
                <li><b>Ensemble Methods:</b> Developed Random Forest and XGBoost models for better non-linear modeling</li>
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
    
    # Key Takeaway 1
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:12px 20px; border-left:6px solid {SECTION_BG}; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0; line-height:1.8;'>
            <b>Baseline Performance:</b> The 70% accuracy serves as a starting point for comparison with more advanced models. 
            This establishes the minimum performance threshold and demonstrates that even simple linear models can capture basic patterns in the data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Takeaway 2
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:12px 20px; border-left:6px solid {SECTION_BG}; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0; line-height:1.8;'>
            <b>Interpretability Advantage:</b> Logistic regression provides clear coefficient values for each feature, 
            making it easy to understand which factors contribute most to hair loss predictions. This transparency is valuable 
            for explaining model decisions to stakeholders and identifying actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Takeaway 3
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:12px 20px; border-left:6px solid {SECTION_BG}; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0; line-height:1.8;'>
            <b>Training Efficiency:</b> With a training time of approximately 2 seconds, this model is ideal for rapid 
            prototyping and iterative development. The fast training allows for quick experimentation with different feature 
            combinations and hyperparameter settings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Takeaway 4
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:12px 20px; border-left:6px solid {SECTION_BG}; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0; line-height:1.8;'>
            <b>Linear Limitations:</b> The model's linear nature fundamentally limits its ability to capture complex 
            non-linear relationships and interactions between features. This constraint explains the 21% performance gap 
            compared to ensemble methods, highlighting the need for more sophisticated algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)