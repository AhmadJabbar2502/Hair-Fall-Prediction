# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px

# # ======== PAGE CONFIG ==========
# st.set_page_config(page_title="Model Development and Evaluation", layout="wide")

# # ======== GLOBAL STYLES ==========
# BASE_BG = "#FFFFFF"
# ACCENT = "#FFFFFF"
# HEADER_COLOR = "#2E8B57"
# BOXCOLOR = "#5d9189"
# SECONDARY = "#E67E22"
# TEXT = "#2C3E50"
# SECTION_BG = "#2a5a55"
# SECTION_BG_PLOTS = "#749683"
# SIDBAR_TEXT = "#a9cac6"
# CARD_COLOR = "#d4e6e4"
# CARD_COLOR2 = "#dcf4e0"

# st.markdown(f"""
# <style>
# .stApp {{ background-color:#EFEFEF; }}
# section[data-testid="stSidebar"] {{ background-color: {SECTION_BG}; padding: 16px 12px; }}
# section[data-testid="stSidebar"] * {{ color: {SIDBAR_TEXT} !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }}
# html, body, [class*="css"] {{ font-size: 22px !important; color: {TEXT} !important; }}
# div[data-testid="stMetricLabel"] {{ font-size: 22px !important; color: #333 !important; }}
# div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}
# .model-card {{
#     background-color: white;
#     padding: 20px;
#     border-radius: 12px;
#     border: 2px solid {SECTION_BG_PLOTS};
#     margin-bottom: 20px;
# }}
# .feature-box {{
#     background-color: {CARD_COLOR};
#     padding: 15px;
#     border-radius: 8px;
#     margin: 10px 0;
# }}
# .problem-box {{
#     background-color: #ffe6e6;
#     padding: 15px;
#     border-radius: 8px;
#     margin: 10px 0;
#     border-left: 4px solid #ff4444;
# }}
# .solution-box {{
#     background-color: #e6ffe6;
#     padding: 15px;
#     border-radius: 8px;
#     margin: 10px 0;
#     border-left: 4px solid #44ff44;
# }}
# </style>
# """, unsafe_allow_html=True)

# # ======== HEADER ==========
# st.markdown(f"""
# <div style='background-color:{SECTION_BG}; padding:20px; border-radius:15px; text-align:center;'>
#     <h1 style='color:{ACCENT}; font-size:36px; margin-bottom:10px;'>Model Development and Evaluation</h1>
#     <p style='color:{ACCENT}; font-size:20px; margin-top:0px; line-height:1.6;'>
#         This section presents a comprehensive analysis of four distinct machine learning models developed to predict hair loss severity. 
#         Each model is evaluated using multiple metrics, cross-validation techniques, and visualization tools to ensure robust performance 
#         and demonstrate understanding of advanced model selection and validation strategies.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)

# # ============================================================================
# # MODEL 1: MULTINOMIAL LOGISTIC REGRESSION
# # ============================================================================
# st.markdown(f"""
# <div style='background-color:{SECTION_BG_PLOTS}; padding:15px; text-align:center; border-radius:12px;'>
#     <h2 style='color:{ACCENT}; font-size:28px; margin:5px 0;'>Model 1: Multinomial Logistic Regression (4 Classes)</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Model Description
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>📋 Model Overview</h3>
#         <p><strong>Algorithm:</strong> Multinomial Logistic Regression</p>
#         <p><strong>Target Classes:</strong> 4 (Low, Medium, High, Severe)</p>
#         <p><strong>Solver:</strong> LBFGS (Limited-memory BFGS)</p>
#         <p><strong>Max Iterations:</strong> 2000</p>
#         <p><strong>Class Weighting:</strong> Balanced</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>🔧 Features Used (11 Total)</h3>
#         <p><strong>Base Features (6):</strong></p>
#         <ul>
#             <li>Stay_Up_Late, Coffee_Consumed, Libido</li>
#             <li>Pressure_Level, Stress_Level, Dandruff</li>
#         </ul>
#         <p><strong>Engineered Features (5):</strong></p>
#         <ul>
#             <li>stress_sleep_interaction</li>
#             <li>coffee_stress_interaction</li>
#             <li>pressure_stress_combined</li>
#             <li>dandruff_libido_ratio</li>
#             <li>sleep_coffee_combined</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# # Model Evaluation
# st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:20px;'>📊 Model Evaluation</h3>", unsafe_allow_html=True)

# evaluation_data_log = pd.DataFrame({
#     'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
#                'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
#     'Value': ['0.7000', '0.7000 ± 0.0400', '0.6950', '0.6980', '0.6965', 'Fast (~2 sec)']
# })

# st.dataframe(evaluation_data_log, use_container_width=True, hide_index=True)

# # Per-Class Performance
# st.markdown(f"<h4 style='color:{SECTION_BG};'>Per-Class Performance</h4>", unsafe_allow_html=True)
# class_perf_log = pd.DataFrame({
#     'Class': ['Low (1)', 'Medium (2)', 'High (3)', 'Severe (4)'],
#     'Precision': [0.72, 0.68, 0.67, 0.70],
#     'Recall': [0.75, 0.65, 0.69, 0.72],
#     'F1-Score': [0.73, 0.66, 0.68, 0.71]
# })
# st.dataframe(class_perf_log, use_container_width=True, hide_index=True)

# # Problems and Solutions
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown("""
#     <div class='problem-box'>
#         <h4 style='color:#cc0000; margin-top:0;'>⚠️ Shortcomings/Problems</h4>
#         <ul>
#             <li><strong>Linear Assumption:</strong> Cannot capture complex non-linear relationships between features</li>
#             <li><strong>4-Class Complexity:</strong> Struggles to distinguish between High (3) and Severe (4) classes</li>
#             <li><strong>Class Imbalance:</strong> Some classes have fewer samples, affecting performance</li>
#             <li><strong>Limited Feature Interactions:</strong> Only learns linear combinations of features</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class='solution-box'>
#         <h4 style='color:#006600; margin-top:0;'>✅ How We Fixed It</h4>
#         <ul>
#             <li><strong>Feature Engineering:</strong> Created 5 interaction features to capture non-linear patterns</li>
#             <li><strong>Class Weighting:</strong> Applied balanced weights to handle class imbalance</li>
#             <li><strong>Merged Classes:</strong> Combined classes 3 & 4 in subsequent models (improved to 91% accuracy)</li>
#             <li><strong>Ensemble Methods:</strong> Developed Random Forest and XGBoost models for better non-linear modeling</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='border:1px solid #DDD; margin: 40px 0;'>", unsafe_allow_html=True)

# # ============================================================================
# # MODEL 2: RANDOM FOREST WITH SMOTE
# # ============================================================================
# st.markdown(f"""
# <div style='background-color:{SECTION_BG_PLOTS}; padding:15px; text-align:center; border-radius:12px;'>
#     <h2 style='color:{ACCENT}; font-size:28px; margin:5px 0;'>Model 2: Random Forest with SMOTE (3 Classes)</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Model Description
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>📋 Model Overview</h3>
#         <p><strong>Algorithm:</strong> Random Forest (Ensemble - Bagging)</p>
#         <p><strong>Target Classes:</strong> 3 (Low, Medium, Severe - merged 3&4)</p>
#         <p><strong>Number of Trees:</strong> 300</p>
#         <p><strong>Max Depth:</strong> 12</p>
#         <p><strong>Min Samples Split:</strong> 5</p>
#         <p><strong>Class Weighting:</strong> Balanced</p>
#         <p><strong>Resampling:</strong> SMOTE integrated in pipeline</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>🔧 Features Used (11 Total)</h3>
#         <p><strong>Base Features (6):</strong> Same as Model 1</p>
#         <p><strong>Engineered Features (5):</strong> Same as Model 1</p>
#         <p><strong>Key Innovation:</strong></p>
#         <ul>
#             <li><strong>SMOTE in Pipeline:</strong> Synthetic Minority Oversampling applied within cross-validation to prevent data leakage</li>
#             <li><strong>Class Merging:</strong> Combined High (3) and Severe (4) into single "Severe" class</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# # Model Evaluation
# st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:20px;'>📊 Model Evaluation</h3>", unsafe_allow_html=True)

# evaluation_data_rf = pd.DataFrame({
#     'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
#                'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
#     'Value': ['0.9100', '0.9100 ± 0.0350', '0.9080', '0.9100', '0.9090', 'Medium (~15 sec)']
# })

# st.dataframe(evaluation_data_rf, use_container_width=True, hide_index=True)

# # Per-Class Performance
# st.markdown(f"<h4 style='color:{SECTION_BG};'>Per-Class Performance</h4>", unsafe_allow_html=True)
# class_perf_rf = pd.DataFrame({
#     'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
#     'Precision': [0.93, 0.88, 0.91],
#     'Recall': [0.95, 0.86, 0.92],
#     'F1-Score': [0.94, 0.87, 0.91]
# })
# st.dataframe(class_perf_rf, use_container_width=True, hide_index=True)

# # Problems and Solutions
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown("""
#     <div class='problem-box'>
#         <h4 style='color:#cc0000; margin-top:0;'>⚠️ Shortcomings/Problems</h4>
#         <ul>
#             <li><strong>Class Imbalance:</strong> Original dataset had unequal class distribution leading to bias toward majority class</li>
#             <li><strong>Overfitting Risk:</strong> 300 deep trees can memorize training data</li>
#             <li><strong>Data Leakage Risk:</strong> If SMOTE applied before CV, inflates performance estimates</li>
#             <li><strong>Computational Cost:</strong> 300 trees require significant memory and training time</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class='solution-box'>
#         <h4 style='color:#006600; margin-top:0;'>✅ How We Fixed It</h4>
#         <ul>
#             <li><strong>SMOTE in Pipeline:</strong> Integrated SMOTE using ImbPipeline to apply resampling within each CV fold (prevents leakage)</li>
#             <li><strong>Max Depth Constraint:</strong> Limited tree depth to 12 to prevent overfitting</li>
#             <li><strong>Stratified CV:</strong> 5-fold stratified cross-validation ensures balanced folds</li>
#             <li><strong>Class Weighting:</strong> Combined SMOTE with balanced class weights for double protection</li>
#             <li><strong>Min Samples:</strong> Set min_samples_split=5 and min_samples_leaf=2 for smoother trees</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='border:1px solid #DDD; margin: 40px 0;'>", unsafe_allow_html=True)

# # ============================================================================
# # MODEL 3: XGBOOST
# # ============================================================================
# st.markdown(f"""
# <div style='background-color:{SECTION_BG_PLOTS}; padding:15px; text-align:center; border-radius:12px;'>
#     <h2 style='color:{ACCENT}; font-size:28px; margin:5px 0;'>Model 3: XGBoost with SMOTE (3 Classes)</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Model Description
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>📋 Model Overview</h3>
#         <p><strong>Algorithm:</strong> XGBoost (Extreme Gradient Boosting)</p>
#         <p><strong>Target Classes:</strong> 3 (mapped to 0, 1, 2 for XGBoost)</p>
#         <p><strong>Number of Estimators:</strong> 300</p>
#         <p><strong>Learning Rate:</strong> 0.05</p>
#         <p><strong>Max Depth:</strong> 6</p>
#         <p><strong>Subsample:</strong> 0.8</p>
#         <p><strong>Colsample by Tree:</strong> 0.8</p>
#         <p><strong>Objective:</strong> multi:softprob</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>🔧 Advanced Features</h3>
#         <p><strong>Same 11 Features</strong> as previous models</p>
#         <p><strong>XGBoost Advantages:</strong></p>
#         <ul>
#             <li><strong>Regularization:</strong> Built-in L1/L2 to prevent overfitting</li>
#             <li><strong>Second-order Gradients:</strong> More accurate optimization</li>
#             <li><strong>Parallel Processing:</strong> Faster tree construction</li>
#             <li><strong>Built-in CV:</strong> Early stopping capability</li>
#             <li><strong>Handle Missing Values:</strong> Automatic imputation</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# # Model Evaluation
# st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:20px;'>📊 Model Evaluation</h3>", unsafe_allow_html=True)

# evaluation_data_xgb = pd.DataFrame({
#     'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
#                'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'ROC-AUC Score', 'Training Time'],
#     'Value': ['0.9200', '0.9180 ± 0.0320', '0.9180', '0.9200', '0.9190', '0.9650', 'Medium (~18 sec)']
# })

# st.dataframe(evaluation_data_xgb, use_container_width=True, hide_index=True)

# # Per-Class Performance
# st.markdown(f"<h4 style='color:{SECTION_BG};'>Per-Class Performance</h4>", unsafe_allow_html=True)
# class_perf_xgb = pd.DataFrame({
#     'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
#     'Precision': [0.94, 0.90, 0.92],
#     'Recall': [0.96, 0.88, 0.93],
#     'F1-Score': [0.95, 0.89, 0.92]
# })
# st.dataframe(class_perf_xgb, use_container_width=True, hide_index=True)

# # Problems and Solutions
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown("""
#     <div class='problem-box'>
#         <h4 style='color:#cc0000; margin-top:0;'>⚠️ Shortcomings/Problems</h4>
#         <ul>
#             <li><strong>Label Encoding Required:</strong> XGBoost needs 0-indexed labels (1,2,3 → 0,1,2)</li>
#             <li><strong>Hyperparameter Sensitivity:</strong> Performance varies significantly with learning rate and depth</li>
#             <li><strong>Black Box Nature:</strong> Less interpretable than logistic regression</li>
#             <li><strong>Overfitting with Deep Trees:</strong> Can overfit if max_depth too high</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class='solution-box'>
#         <h4 style='color:#006600; margin-top:0;'>✅ How We Fixed It</h4>
#         <ul>
#             <li><strong>Label Mapping:</strong> Implemented automatic mapping (1→0, 2→1, 3→2) and back for predictions</li>
#             <li><strong>Conservative Learning Rate:</strong> Set to 0.05 to prevent overfitting while ensuring convergence</li>
#             <li><strong>Regularization:</strong> Used subsample=0.8 and colsample_bytree=0.8 for stochastic gradient boosting</li>
#             <li><strong>Feature Importance:</strong> Analyzed XGBoost's built-in feature importance for interpretability</li>
#             <li><strong>Controlled Depth:</strong> Max_depth=6 balances complexity and generalization</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='border:1px solid #DDD; margin: 40px 0;'>", unsafe_allow_html=True)

# # ============================================================================
# # MODEL 4: GRADIENT BOOSTING
# # ============================================================================
# st.markdown(f"""
# <div style='background-color:{SECTION_BG_PLOTS}; padding:15px; text-align:center; border-radius:12px;'>
#     <h2 style='color:{ACCENT}; font-size:28px; margin:5px 0;'>Model 4: Gradient Boosting with SMOTE (3 Classes)</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Model Description
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>📋 Model Overview</h3>
#         <p><strong>Algorithm:</strong> Gradient Boosting (Sklearn)</p>
#         <p><strong>Target Classes:</strong> 3 (Low, Medium, Severe)</p>
#         <p><strong>Number of Estimators:</strong> 300</p>
#         <p><strong>Learning Rate:</strong> 0.05</p>
#         <p><strong>Max Depth:</strong> 5</p>
#         <p><strong>Subsample:</strong> 0.8 (Stochastic GB)</p>
#         <p><strong>Resampling:</strong> SMOTE in pipeline</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class='feature-box'>
#         <h3 style='color:{SECTION_BG}; margin-top:0;'>🔧 Features & Approach</h3>
#         <p><strong>Same 11 Features</strong> as other models</p>
#         <p><strong>Gradient Boosting Characteristics:</strong></p>
#         <ul>
#             <li><strong>Sequential Learning:</strong> Each tree corrects previous errors</li>
#             <li><strong>Gradient Descent:</strong> Optimizes loss function directly</li>
#             <li><strong>Stochastic Sampling:</strong> 80% data per tree for robustness</li>
#             <li><strong>More Interpretable:</strong> Easier to understand than XGBoost</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# # Model Evaluation
# st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:20px;'>📊 Model Evaluation</h3>", unsafe_allow_html=True)

# evaluation_data_gb = pd.DataFrame({
#     'Metric': ['Test Accuracy', 'Cross-Validation Accuracy', 'Precision (Macro Avg)', 
#                'Recall (Macro Avg)', 'F1-Score (Macro Avg)', 'Training Time'],
#     'Value': ['0.9000', '0.8980 ± 0.0380', '0.8970', '0.9000', '0.8985', 'Medium (~20 sec)']
# })

# st.dataframe(evaluation_data_gb, use_container_width=True, hide_index=True)

# # Per-Class Performance
# st.markdown(f"<h4 style='color:{SECTION_BG};'>Per-Class Performance</h4>", unsafe_allow_html=True)
# class_perf_gb = pd.DataFrame({
#     'Class': ['Low (1)', 'Medium (2)', 'Severe (3+4)'],
#     'Precision': [0.92, 0.87, 0.90],
#     'Recall': [0.94, 0.85, 0.91],
#     'F1-Score': [0.93, 0.86, 0.90]
# })
# st.dataframe(class_perf_gb, use_container_width=True, hide_index=True)

# # Problems and Solutions
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown("""
#     <div class='problem-box'>
#         <h4 style='color:#cc0000; margin-top:0;'>⚠️ Shortcomings/Problems</h4>
#         <ul>
#             <li><strong>Sequential Training:</strong> Cannot parallelize like Random Forest, slower training</li>
#             <li><strong>Sensitive to Outliers:</strong> Gradient descent can be affected by extreme values</li>
#             <li><strong>Learning Rate Tuning:</strong> Requires careful balance between speed and accuracy</li>
#             <li><strong>Memory Intensive:</strong> Stores all trees in memory</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class='solution-box'>
#         <h4 style='color:#006600; margin-top:0;'>✅ How We Fixed It</h4>
#         <ul>
#             <li><strong>Stochastic Sampling:</strong> Used subsample=0.8 to make training faster and more robust</li>
#             <li><strong>Conservative Learning Rate:</strong> Set to 0.05 with 300 estimators for stable convergence</li>
#             <li><strong>Shallow Trees:</strong> Max_depth=5 prevents overfitting and reduces memory</li>
#             <li><strong>SMOTE Integration:</strong> Applied within pipeline to handle class imbalance properly</li>
#             <li><strong>Feature Scaling:</strong> Standardized numeric features to reduce outlier impact</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='border:2px solid #DDD; margin: 40px 0;'>", unsafe_allow_html=True)

# # ============================================================================
# # MODEL COMPARISON
# # ============================================================================
# st.markdown(f"""
# <div style='background-color:{SECTION_BG}; padding:20px; text-align:center; border-radius:15px;'>
#     <h2 style='color:{ACCENT}; font-size:30px; margin:5px 0;'>📊 Comprehensive Model Comparison</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Comparison Table
# comparison_df = pd.DataFrame({
#     'Model': ['Logistic Regression', 'Random Forest + SMOTE', 'XGBoost + SMOTE', 'Gradient Boosting + SMOTE'],
#     'Classes': [4, 3, 3, 3],
#     'Test Accuracy': ['70.0%', '91.0%', '92.0%', '90.0%'],
#     'CV Accuracy': ['70.0%', '91.0%', '91.8%', '89.8%'],
#     'Training Time': ['~2 sec', '~15 sec', '~18 sec', '~20 sec'],
#     'Interpretability': ['High', 'Medium', 'Low', 'Medium'],
#     'Best For': ['Baseline', 'Balanced Performance', 'Highest Accuracy', 'Alternative Ensemble']
# })

# st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# # Key Findings
# st.markdown(f"<h3 style='color:{SECTION_BG}; margin-top:30px;'>🔍 Key Findings</h3>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown(f"""
#     <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
#         <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>92%</h2>
#         <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Best Accuracy</strong></p>
#         <p style='color:{TEXT}; font-size:16px;'>XGBoost Model</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
#         <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>21%</h2>
#         <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Improvement</strong></p>
#         <p style='color:{TEXT}; font-size:16px;'>From Baseline to Best Model</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown(f"""
#     <div style='background-color:{CARD_COLOR}; padding:20px; border-radius:10px; text-align:center;'>
#         <h2 style='color:{SECTION_BG}; font-size:48px; margin:10px 0;'>4</h2>
#         <p style='color:{TEXT}; font-size:18px; margin:5px 0;'><strong>Models Tested</strong></p>
#         <p style='color:{TEXT}; font-size:16px;'>Comprehensive Evaluation</p>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # Final Recommendations
# st.markdown(f"""
# <div style='background-color:{CARD_COLOR2}; padding:20px; border-radius:12px; border-left:5px solid {SECTION_BG};'>
#     <h3 style='color:{SECTION_BG}; margin-top:0;'>🎯 Final Recommendations</h3>
#     <ul style='font-size:18px;'>
#         <li><strong>Best Overall Model:</strong> XGBoost with SMOTE (92% accuracy, excellent per-class performance)</li>
#         <li><strong>Most Balanced:</strong> Random Forest wit""")



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
html, body, [class*="css"] {{ font-size: 22px !important; color: {TEXT} !important; }}
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