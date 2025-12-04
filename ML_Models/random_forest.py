import streamlit as st
import pandas as pd

# Color scheme
BASE_BG = "#FFFFFF"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#749683"
ACCENT = "#FFFFFF"
TEXT = "#2C3E50"
CARD_COLOR = "#d4e6e4"

def render_random_forest_page():
    """Render Random Forest model page"""
    
    # Main Model Heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>
        <h2 style='color:{ACCENT}; font-size:24px; margin:6px 0 6px 0;'>Model 2: Random Forest with SMOTE (3 Classes)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Overview - Simple list format
    st.markdown(f"<h3 style='color:{SECTION_BG}; font-size:20px;'>Model Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='font-size:16px; color:{TEXT}; line-height:1.8;'>
        <li><b>Algorithm:</b> Random Forest (Ensemble - Bagging)</li>
        <li><b>Target Classes:</b> 3 (Low, Medium, Severe - merged 3&4)</li>
        <li><b>Number of Trees:</b> 300</li>
        <li><b>Max Depth:</b> 12</li>
        <li><b>Min Samples Split:</b> 5</li>
        <li><b>Min Samples Leaf:</b> 2</li>
        <li><b>Class Weighting:</b> Balanced</li>
        <li><b>Resampling:</b> SMOTE integrated in pipeline</li>
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
            <h4 style='color:#2C3E50; margin-top:0; font-size:18px;'>SMOTE Integration</h4>
            <p style='color:#2C3E50; font-size:15px; line-height:1.6; margin:0;'>
                Applied Synthetic Minority Oversampling within the pipeline using ImbPipeline, ensuring SMOTE 
                runs independently in each cross-validation fold to prevent data leakage and provide honest performance estimates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:#A8D5BA; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:#2C3E50; margin-top:0; font-size:18px;'>Class Merging Strategy</h4>
            <p style='color:#2C3E50; font-size:15px; line-height:1.6; margin:0;'>
                Combined High (3) and Severe (4) classes into a single "Severe" category, reducing confusion between 
                similar severity levels and improving model accuracy by 21 percentage points over the 4-class baseline.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div style='background-color:#FFB347; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:#2C3E50; margin-top:0; font-size:18px;'>Ensemble Learning</h4>
            <p style='color:#2C3E50; font-size:15px; line-height:1.6; margin:0;'>
                Utilized 300 decision trees with bootstrap aggregating (bagging) to reduce variance and prevent overfitting, 
                where each tree votes on the final prediction creating robust and stable classifications.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background-color:#D9D9D9; padding:20px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-height:180px;'>
            <h4 style='color:#2C3E50; margin-top:0; font-size:18px;'>Regularization Techniques</h4>
            <p style='color:#2C3E50; font-size:15px; line-height:1.6; margin:0;'>
                Implemented max_depth=12, min_samples_split=5, and min_samples_leaf=2 constraints to control tree 
                complexity and create smoother decision boundaries that generalize better to unseen data.
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

    evaluation_data_rf = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
                   'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
        'Value': ['0.9100', '0.9100 ± 0.0350', '0.9080', '0.9100', '0.9090', 'Medium (~15 sec)']
    })

    st.dataframe(evaluation_data_rf, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-Class Performance
    st.markdown(f"<h4 style='color:{SECTION_BG}; font-size:21px;'>Per-Class Performance</h4>", unsafe_allow_html=True)
    class_perf_rf = pd.DataFrame({
        'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
        'Precision': [0.93, 0.88, 0.91],
        'Recall': [0.95, 0.86, 0.92],
        'F1-Score': [0.94, 0.87, 0.91],
        'Support': [45, 52, 53]
    })
    st.dataframe(class_perf_rf, use_container_width=True, hide_index=True)

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
                <li><b>Class Imbalance:</b> Original dataset had unequal class distribution leading to bias toward majority class</li>
                <li><b>Overfitting Risk:</b> 300 deep trees can memorize training data patterns instead of learning generalizable rules</li>
                <li><b>Data Leakage Risk:</b> If SMOTE applied before CV, synthetic samples leak into test folds, inflating performance</li>
                <li><b>Computational Cost:</b> 300 trees with depth 12 require significant memory and training time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#e6ffe6; padding:15px; border-radius:8px; border-left:4px solid #44ff44;'>
            <h4 style='color:#006600; margin-top:0; font-size:18px;'>How I Fixed It</h4>
            <ul style='font-size:16px;'>
                <li><b>SMOTE in Pipeline:</b> Integrated SMOTE using ImbPipeline to apply resampling within each CV fold preventing leakage</li>
                <li><b>Max Depth Constraint:</b> Limited tree depth to 12 to prevent overfitting while maintaining expressiveness</li>
                <li><b>Stratified CV:</b> 5-fold stratified cross-validation ensures each fold maintains original class distribution</li>
                <li><b>Double Protection:</b> Combined SMOTE resampling with class_weight='balanced' for additional minority class emphasis</li>
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
            <li><b>Major Improvement:</b> 21% accuracy gain over logistic regression baseline (70% to 91%)</li>
            <li><b>Balanced Performance:</b> Strong F1 scores across all three classes (Low=0.94, Medium=0.87, Severe=0.91)</li>
            <li><b>SMOTE Integration:</b> Pipeline architecture prevents data leakage while handling class imbalance effectively</li>
            <li><b>Feature Engineering Impact:</b> Engineered interaction features contribute 35% of total feature importance</li>
            <li><b>Stable Predictions:</b> Low CV standard deviation (±0.035) indicates consistent performance across folds</li>
            <li><b>Practical Choice:</b> Best balance of accuracy, interpretability through feature importance, and reliability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)