import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from src.missingness import impute_medical_condition_mode
from src.heatmaps import plot_missingness_heatmap_nutritional_deficiencies

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Missingness Analysis", layout="wide")

# ======== GLOBAL STYLES ==========
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
SIDBAR_TEXT = "#a9cac6"

st.markdown("""
    <style>
    /* Change the main app background */
    .stApp {
        background-color:#EFEFEF; /* keep existing background */
    }

    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #2a5a55;  /* sidebar background color */
        padding: 16px 12px;
    }

    /* --- SIDEBAR TEXT --- */
    section[data-testid="stSidebar"] * {
        color: #a9cac6 !important;   /* keep your text color */
        font-size: 16px !important;  /* match Hair Baldness Story page */
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* General text styling */
    html, body, [class*="css"] {
        font-size: 18px !important; /* match Hair Baldness Story page */
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 18px !important;
        color: #333 !important;
    }

    /* DataFrame table styling remains the same */
    div[data-testid="stDataFrame"] {
        background-color: white !important;
        border: 3px solid #2a5a55 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    </style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{BASE_BG}; padding:15px; border-radius:40px; text-align:center;'>
    <h1 style='color:#5e928a; font-size:25px; margin-bottom:5px;'>Missingness Analysis</h1>
    <p style='color:{HEADER_COLOR}; font-size:20px; margin-top:0px;'>
        Investigate patterns of missing data and visualize their relationships with key features.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

# ======== LOAD DATASETS ==========
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

df_predict_raw = load_csv("Data/Predict Hair Fall Raw.csv")
df_predict_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df_luke_raw = load_csv("Data/Luke_hair_loss_documentation Raw.csv")
df_luke_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

# ======== DATASET SELECTOR ==========
dataset_option = st.selectbox(
    "Select Dataset:",
    ["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset"],
    index=0
)

if dataset_option == "None":
    st.info("Select a dataset from the dropdown above to begin.")
else:
    # ===== Dataset Banner =====
    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:12px;'>"
        f"<h2 style='color:{ACCENT}; font-size:25px; margin:5px 0 5px 0;'>{dataset_option}</h2></div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ===================== Hair Health Prediction Dataset =====================
    if dataset_option == "Hair Health Prediction Dataset":
        if df_predict_raw is None:
            st.error("Raw data not found.")
        else:
            hair_raw = df_predict_raw.copy()

            # ---- Heatmap for missing values ----
            subset_cols = ["Medical_Conditions", "Medications_and_Treatments", "Nutritional_Deficiencies"]
            subset_cols = [c for c in subset_cols if c in hair_raw.columns]

            if subset_cols:
                missing_matrix = hair_raw[subset_cols].isna().astype(int).to_numpy()
                fig, ax = plt.subplots(figsize=(11, 5))
                im = ax.imshow(missing_matrix.T, interpolation="nearest", aspect="auto", cmap="YlGn")
                ax.set_xlabel("Index", fontsize=13, color=TEXT)
                ax.set_ylabel("Features", fontsize=13, color=TEXT)
                ax.set_yticks(range(len(subset_cols)))
                ax.set_yticklabels(subset_cols, fontsize=12, color=TEXT)
                ax.set_title("Missing Values Heatmap", fontsize=16, fontweight="600", color=HEADER_COLOR)
                ax.grid(True, axis="y", linestyle="--", alpha=0.6)
                fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
                st.pyplot(fig)
                plt.close(fig)

                # ---- Missing counts table ----
                st.markdown(f"<p style='font-size:25px; color:{TEXT}; text-align:center; '><b>Missing Value Counts</b></p>", unsafe_allow_html=True)
                missing_counts = hair_raw.isna().sum().reset_index().rename(columns={"index":"Feature", 0:"Missing Count"})
                st.dataframe(missing_counts, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingness in Medical Conditions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # ---- Observation Card ----
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
                <p style='font-size:18px; color:{TEXT}; margin:0;'>
                <b>Observation:</b> Missingness in <b>Medical_Conditions</b> is not completely random (not MCAR). Chi-squared tests reveal a strong dependency on <b>Stress Level</b> and <b>Age Range</b>. Higher stress and older age correlate with more missing entries. Thus, these values are MAR.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ---- Encode & sort for visualizations ----
            if "Stress" in hair_raw.columns and "Stress_Level" not in hair_raw.columns:
                hair_raw['Stress_Level'] = hair_raw['Stress']
            if "Age" in hair_raw.columns and "Age_Range" not in hair_raw.columns:
                hair_raw['Age_Range'] = pd.cut(hair_raw['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
            hair_raw["Medical_Conditions_missing"] = hair_raw["Medical_Conditions"].isna().astype(int)

            # ---- Missingness vs Stress and Age Heatmaps ----
            st.markdown(f"<p style='font-size:22px; color:{TEXT}; text-align:center;'><b>Missingness in Medical Conditions by Stress Level & Age Range</b></p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")

            # Heatmap by Stress
            with col1:
                stress_order = ["Low", "Moderate", "High"]
                hair_raw["Stress_Level"] = pd.Categorical(hair_raw["Stress_Level"], categories=stress_order, ordered=True)
                hair_sorted = hair_raw.sort_values("Stress_Level")
                matrix_stress = hair_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)
                fig, ax = plt.subplots(figsize=(10,2))
                sns.heatmap(matrix_stress, cmap="YlGn", cbar=True, ax=ax)
                xticks = [np.mean(np.where(hair_sorted["Stress_Level"] == lvl)) for lvl in stress_order if lvl in hair_sorted["Stress_Level"].unique()]
                ax.set_xticks(xticks)
                ax.set_xticklabels([lvl for lvl in stress_order if lvl in hair_sorted["Stress_Level"].unique()], fontsize=11, rotation=0)
                ax.set_yticks([0])
                ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
                ax.set_title("By Stress Level", fontsize=14, color=HEADER_COLOR)
                st.pyplot(fig)
                plt.close(fig)

            # Heatmap by Age
            with col2:
                age_sorted = hair_raw.sort_values("Age_Range", na_position='last')
                matrix_age = age_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)
                fig, ax = plt.subplots(figsize=(10,2))
                sns.heatmap(matrix_age, cmap="YlGn", cbar=True, ax=ax)
                uniq_age = list(age_sorted["Age_Range"].dropna().unique())
                xticks = [np.mean(np.where(age_sorted["Age_Range"] == lvl)) for lvl in uniq_age]
                ax.set_xticks(xticks)
                ax.set_xticklabels(uniq_age, fontsize=11, rotation=0)
                ax.set_yticks([0])
                ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
                ax.set_title("By Age Range", fontsize=14, color=HEADER_COLOR)
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            
            # ---- Chi-Squared Tests Summary ----
            st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Chi-Squared Tests for Missingness</h2>", unsafe_allow_html=True)
            st.markdown(
                """
                <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
                The missingness in <b>Medical Conditions</b> was tested against both <b>Stress Level</b> and <b>Age Range</b>.<br>
                The analysis confirms that missingness is <b>not random</b>.<br><br>
                - Probability that the difference between the observed and expected value of missingness due to Stress Level by chance: <b>7.756e-17</b><br>
                - Probability that the difference between the observed and expected value of missingness due to Age Range by chance: <b>2.335e-23</b><br><br>
                These extremely low probabilities indicate a strong relationship between missingness and these variables.
                </p>
                """.replace("{TEXT}", TEXT),
                unsafe_allow_html=True
            )


           # Contingency tables & chi-squared
            contingency_stress = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Stress_Level'])
            contingency_age = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Age_Range'])
            chi2_stress, p_stress, dof_stress, ex_stress = stats.chi2_contingency(contingency_stress)
            chi2_age, p_age, dof_age, ex_age = stats.chi2_contingency(contingency_age)

            x = np.arange(len(contingency_stress.columns))
            width = 0.35

            col1, col2 = st.columns(2, gap="large")

            # Stress Level plot
            with col1:
                fig, ax = plt.subplots(figsize=(8,5))
                obs = contingency_stress.loc[1]
                exp = pd.Series(ex_stress[1], index=contingency_stress.columns)
                ax.bar(x - width/2, obs, width, label='Observed', color='#86aca9')
                ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
                ax.set_xticks(x)
                ax.set_xticklabels(obs.index)
                ax.set_title("Missingness: Stress Level", fontsize=14, color=HEADER_COLOR)
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            # Age Range plot
            with col2:
                fig, ax = plt.subplots(figsize=(8,5))
                obs = contingency_age.loc[1]
                exp = pd.Series(ex_age[1], index=contingency_age.columns)
                ax.bar(x - width/2, obs, width, label='Observed', color='#A8D5BA')
                ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
                ax.set_xticks(x)
                ax.set_xticklabels(obs.index)
                ax.set_title("Missingness: Age Range", fontsize=14, color=HEADER_COLOR)
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            
            # Logistic Regression section
            st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Logistic Regression for Missingness in Medical Conditions</h2>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
                Moreover, we perform a <b>logistic regression</b> to evaluate the relationship between <b>Stress Level</b> and the probability 
                of missing values in <b>Medical Conditions</b>. This approach allows us to quantify how strongly stress levels can predict missingness.
                </p>
                """,
                unsafe_allow_html=True
            )

            # Logistic regression table (static representation)
            st.markdown(
                f"""
                <table style="width:70%; border-collapse:collapse; font-size:16px; color:{TEXT}; margin-left:auto; margin-right:auto;">
                    <tr style="background-color:#f2f2f2;">
                        <th> </th><th>coef</th><th>std err</th><th>z</th><th>P>|z|</th><th>[0.025</th><th>0.975]</th>
                    </tr>
                    <tr><td>const</td><td>-2.7795</td><td>0.247</td><td>-11.264</td><td>0.000</td><td>-3.263</td><td>-2.296</td></tr>
                    <tr><td>Weight_Loss_Encoding</td><td>0.0817</td><td>0.204</td><td>0.401</td><td>0.688</td><td>-0.317</td><td>0.481</td></tr>
                    <tr><td>Genetic_Encoding</td><td>0.3407</td><td>0.207</td><td>1.649</td><td>0.099</td><td>-0.064</td><td>0.746</td></tr>
                    <tr><td>Stress_Level</td><td>0.4212</td><td>0.130</td><td>3.243</td><td>0.001</td><td>0.167</td><td>0.676</td></tr>
                </table>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
                We can observe that the <b>p-value for Stress_Level</b> is very low (0.001), indicating that it is statistically significant. 
                This confirms that <b>Stress Level</b> has a meaningful relationship with the missingness of <b>Medical Conditions</b>.
                </p>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


            # ===== Imputation Info =====
            
            # ---- Mode-based Imputation Section ----
            df_raw = pd.read_csv("Data/Predict Hair Fall Raw.csv")
            df_cleaned = pd.read_csv("Data/Predict Hair Fall Cleaned.csv")
            
            st.markdown(
                f"""
                <div style='background-color:#C7E9C0; padding:12px; border-radius:12px; text-align:center;'>
                    <h2 style='font-size:25px; color:#2E8B57; margin:0;'>Imputation of Missing Values </h2>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >First Method: Mode (Conditional)</h2>", unsafe_allow_html=True)

            st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
            In addition to Random Forest, missing <b>Medical_Conditions</b> can be imputed using a <b>conditional mode</b> approach. 
            Here, each missing value is filled with the most frequent condition observed within its subgroup defined by <b>Age Range</b> 
            and <b>Stress Level</b>. This method preserves the underlying structure and distribution of the data across different groups.
            </p>
            """, unsafe_allow_html=True)

            # ---- Apply Mode-based Imputation ----
            df_mode = df_raw.copy()
            bins = [18, 30, 40, 51]
            labels = ['18-30', '30-40', '40-51']
            df_mode['Age_Range'] = pd.cut(df_mode['Age'], bins=bins, labels=labels, right=False)
            df_mode['Stress_Level'] = df_mode['Stress'].map({'Low': 0, 'Moderate': 1, 'High': 2})
            df_mode['Medical_Conditions'] = df_mode.apply(lambda row: impute_medical_condition_mode(row, df_mode), axis=1)

            # ---- Before vs After Mode Imputation Plot ----
            count_mode = df_mode['Medical_Conditions'].value_counts().sort_index()
            count_raw = df_raw['Medical_Conditions'].value_counts().sort_index()

            all_conditions = sorted(set(count_raw.index).union(set(count_mode.index)))
            count_raw = count_raw.reindex(all_conditions, fill_value=0)
            count_mode = count_mode.reindex(all_conditions, fill_value=0)

            fig, ax = plt.subplots(figsize=(12,6))
            x = range(len(all_conditions))
            width = 0.35

            ax.bar([i - width/2 for i in x], count_raw.values, width, label='Before Imputation', color='#86aca9')
            ax.bar([i + width/2 for i in x], count_mode.values, width, label='After Mode Imputation', color='#5d9189')

            ax.set_xticks(x)
            ax.set_xticklabels(all_conditions, rotation=45, ha='right')
            ax.set_ylabel('Count', fontsize=14)
            ax.set_title('Medical Conditions: Before vs After Mode Imputation', fontsize=16, color='#2E8B57')
            ax.legend(fontsize=12)
            ax.tick_params(axis='y', labelsize=12)

            st.pyplot(fig)
            plt.close(fig)
            # --- RANDOM FOREST IMPUTATION ----
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >Second Method: Random Forest</h2>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
            Missing <b>Medical_Conditions</b> were imputed using <b>Random Forest (n_estimators=1000)</b> 
            trained on features: Stress Level, Age Range, Genetic Encoding, Hormonal Changes, Smoking, Weight Loss, Environmental Factors.
            </p>
            """, unsafe_allow_html=True)

            # ---- Before vs After Imputation Plot ----

            df_raw['Medical_Conditions'] = df_raw['Medical_Conditions'].dropna().astype(str).str.strip().str.title()
            df_cleaned['Medical_Conditions'] = df_cleaned['Medical_Conditions'].dropna().astype(str).str.strip().str.title()

            count_raw = df_raw['Medical_Conditions'].value_counts().sort_index()
            count_cleaned = df_cleaned['Medical_Conditions'].value_counts().sort_index()
            all_conditions = sorted(set(count_raw.index).union(set(count_cleaned.index)))
            count_raw = count_raw.reindex(all_conditions, fill_value=0)
            count_cleaned = count_cleaned.reindex(all_conditions, fill_value=0)

            fig, ax = plt.subplots(figsize=(12,6))
            x = range(len(all_conditions))
            width = 0.35
            ax.bar([i - width/2 for i in x], count_raw.values, width, label='Before Imputation', color='#86aca9')
            ax.bar([i + width/2 for i in x], count_cleaned.values, width, label='After Imputation', color='#5d9189')
            ax.set_xticks(x)
            ax.set_xticklabels(all_conditions, rotation=45, ha='right')
            ax.set_ylabel('Count', fontsize=14)
            ax.set_title('Medical Conditions: Before vs After Imputation', fontsize=16, color='#2E8B57')
            ax.legend(fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown(f"""
            <div style='background-color:#FFF4E5; padding:14px; border-radius:12px;'>
                <p style='font-size:18px; color:{TEXT}; margin:0;'>
                <b>Observation:</b> In this scenario, <b>Random Forest imputation</b> is more suitable than <b>Mode imputation</b>. 
                For example, in the group with <b>Stress Level = 2</b> and <b>Age Range = 18-30</b>, there is no single dominant <b>Medical_Condition</b>; counts are very close (e.g., 9 or 8 for each condition). 
                Mode imputation would arbitrarily assign the most frequent value, potentially introducing bias. 
                Random Forest can leverage multiple features simultaneously to make more informed predictions, preserving the underlying distribution.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingess in Nutritional Deficiencies</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # ---- Observation Card ----
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
                <p style='font-size:18px; color:{TEXT}; margin:0;'>
                <b>Observation:</b> Missingness in <b>Nutritional Deficiencies</b> is not completely random (not MCAR). Chi-squared tests reveal a strong dependency on <b>Age Range</b>. Less age (18-30) correlate with all the missing entries. Thus, these values are MAR.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"<p style='font-size:22px; color:{TEXT}; text-align:center;'><b>Missingness in Nutritional Deficiencies by  Age Range</b></p>", unsafe_allow_html=True)
            hair_raw['Nutritional_Deficiencies_missing'] = hair_raw['Nutritional_Deficiencies'].isna().astype(int)
            age_sorted = hair_raw.sort_values('Age_Range')
            missing_age_matrix = age_sorted['Nutritional_Deficiencies_missing'].to_numpy().reshape(1, -1)
            
            matrix_age = age_sorted["Nutritional_Deficiencies_missing"].to_numpy().reshape(1, -1)
            fig, ax = plt.subplots(figsize=(10,2))
            sns.heatmap(matrix_age, cmap="YlGn", cbar=True, ax=ax)
            uniq_age = list(age_sorted["Age_Range"].dropna().unique())
            xticks = [np.mean(np.where(age_sorted["Age_Range"] == lvl)) for lvl in uniq_age]
            ax.set_xticks(xticks)
            ax.set_xticklabels(uniq_age, fontsize=11, rotation=0)
            ax.set_yticks([0])
            ax.set_yticklabels(["Nutritional_Deficiencies_missing"], fontsize=11)
            ax.set_title("By Age Range", fontsize=14, color=HEADER_COLOR)
            st.pyplot(fig)
            plt.close(fig)
            
            
            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            
            # ---- Chi-Squared Tests Summary ----
            st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Chi-Squared Tests for Missingness</h2>", unsafe_allow_html=True)
            st.markdown(
                """
                <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
                The missingness in <b>Nutritional Deficiencies</b> was tested against <b>Age Range</b>. The analysis confirms that missingness is <b>not random</b>.Probability that the difference between the observed and expected value of missingness due to Age Range by chance: <b>4.725e-36</b><br><br>
                These extremely low probabilities indicate a strong relationship between missingness and these variables.
                </p>
                """.replace("{TEXT}", TEXT),
                unsafe_allow_html=True
            )
            
            contingency_age = pd.crosstab(hair_raw['Nutritional_Deficiencies_missing'], hair_raw['Age_Range'])
            chi2_stress, p_stress, dof_stress, ex_stress = stats.chi2_contingency(contingency_stress)
            chi2_age, p_age, dof_age, ex_age = stats.chi2_contingency(contingency_age)

            x = np.arange(len(contingency_stress.columns))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8,5))
            obs = contingency_age.loc[1]
            exp = pd.Series(ex_age[1], index=contingency_age.columns)
            ax.bar(x - width/2, obs, width, label='Observed', color='#A8D5BA')
            ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
            ax.set_xticks(x)
            ax.set_xticklabels(obs.index)
            ax.set_title("Missingness: Age Range", fontsize=10, color=HEADER_COLOR)
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            
            # Logistic Regression section for Nutritional_Deficiencies
            st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Logistic Regression for Missingness in Nutritional Deficiencies</h2>", unsafe_allow_html=True)

            st.markdown(
            f"""
            <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
            We perform a <b>logistic regression</b> to assess how <b>Age</b> relates to the probability of missing values 
            in <b>Nutritional_Deficiencies</b>. This analysis helps identify which factors significantly influence missingness.
            </p>
            """,
            unsafe_allow_html=True
            )

            # Logistic regression table (static representation)
            st.markdown(
            f"""
            <table style="width:70%; border-collapse:collapse; font-size:16px; color:{TEXT}; margin-left:auto; margin-right:auto;">
            <tr style="background-color:#f2f2f2;">
                <th> </th><th>coef</th><th>std err</th><th>z</th><th>P>|z|</th><th>[0.025</th><th>0.975]</th>
            </tr>
            <tr><td>const</td><td>4.7461</td><td>0.711</td><td>6.679</td><td>0.000</td><td>3.353</td><td>6.139</td></tr>
            <tr><td>Weight_Loss_Encoding</td><td>-0.1755</td><td>0.269</td><td>-0.653</td><td>0.514</td><td>-0.703</td><td>0.351</td></tr>
            <tr><td>Genetic_Encoding</td><td>0.1333</td><td>0.269</td><td>0.496</td><td>0.620</td><td>-0.394</td><td>0.660</td></tr>
            <tr><td>Age</td><td>-0.2659</td><td>0.029</td><td>-9.099</td><td>0.002</td><td>-0.323</td><td>-0.209</td></tr>
            </table>
            """,
            unsafe_allow_html=True
            )

            st.markdown(
            f"""
            <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
            We can observe that the <b>p-value for Age</b> is very low (0.002), indicating that it is statistically significant. 
            This confirms that <b>Age</b> has a meaningful relationship with the missingness of <b>Nutritional_Deficiencies</b>.
            </p>
            """,
            unsafe_allow_html=True
            )

            st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


    else:
        # ===== Missing Values Analysis =====
        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Missing Values Heatmap ----
        subset_cols = ['School_Assesssment', 'Hair_Grease', 'Dandruff']

        if subset_cols:
            missing_matrix = df_luke_raw[subset_cols].isna().astype(int).to_numpy()
            fig, ax = plt.subplots(figsize=(11, 5))
            im = ax.imshow(missing_matrix.T, interpolation="nearest", aspect="auto", cmap="YlGn")
            ax.set_xlabel("Index", fontsize=13, color=TEXT)
            ax.set_ylabel("Features", fontsize=13, color=TEXT)
            ax.set_yticks(range(len(subset_cols)))
            ax.set_yticklabels(subset_cols, fontsize=12, color=TEXT)
            ax.set_title("Missing Values Heatmap", fontsize=16, fontweight="600", color=HEADER_COLOR)
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            st.pyplot(fig)
            plt.close(fig)

        # ---- Missing Value Counts ----
        st.markdown(f"<p style='font-size:23px; color:{TEXT}; text-align:center; '><b>Missing Value Counts</b></p>", unsafe_allow_html=True)
        missing_counts = df_luke_raw.isna().sum().reset_index()
        missing_counts.columns = ["Column", "Missing Values"]
        missing_counts["Missing Percentage"] = (missing_counts["Missing Values"] / len(df_luke_raw) * 100).round(2)
        st.dataframe(
            missing_counts.style.set_properties(**{'font-size': '20px'})
        )
        
        # --- Missingess in School Assessment and Dandruff --- #
        st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingess in School Assessment and Dandruff</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT};'>
            Although the missingness table highlights several null entries, these do not represent true missing data. 
            Based on domain understanding and logical inference, the following replacements and actions were taken:
            </p>
            <ul style='font-size:18px; color:{TEXT};'>
            <li><b>School_Assessment:</b> Missing entries indicate that no assignments were given, and were therefore replaced with <i>'No Assessment'</i>.</li>
            <li><b>Dandruff:</b> Missing values suggest the absence of dandruff, and were replaced with <i>'None'</i>.</li>
            </ul>
            """, unsafe_allow_html=True)
        
        # --- Missingess in Hair Grease --- #
        st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingess in Hair Grease</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT};'>
            In the <b>Hair_Grease</b> column, only four missing values were detected. 
            To handle these, two approaches were explored to assess their effect on the dataset:
            </p>
            <ul style='font-size:18px; color:{TEXT};'>
            <li><b>Approach 1 — Dropping Nulls:</b> The four missing entries were removed to maintain data integrity without introducing artificial estimates.</li>
            <li><b>Approach 2 — Mean Imputation:</b> The missing values were replaced with the column mean to observe whether this altered the correlation between <b>Hair_Grease</b> and other variables.</li>
            </ul>
            """, unsafe_allow_html=True)
        
        
        # Select numeric columns only
        numeric_cols = df_luke_raw.select_dtypes(include=np.number).columns.tolist()
        df_luke_raw['Hair_Loss_Encoding'] = df_luke_raw['Hair_Loss'].map({'Few': 1, 'Medium': 2, 'Many': 3, 'A lot': 4 })
        df_luke_raw['Pressure_Level_Encoding'] = df_luke_raw['Pressure_Level'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4 })
        df_luke_raw['Stress_Level_Encoding'] = df_luke_raw['Stress_Level'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4 })
        numeric_cols = numeric_cols + ['Stress_Level_Encoding', 'Hair_Loss_Encoding', 'Pressure_Level_Encoding'] 

        # 1️⃣ Original dataset
        corr_original = df_luke_raw[numeric_cols].corr()
        hair_grease_corr_original = corr_original.loc['Hair_Grease'].drop('Hair_Grease')

        # 2️⃣ Dropping null values
        df_drop = df_luke_raw.dropna(subset=['Hair_Grease'])
        corr_drop = df_drop[numeric_cols].corr()
        hair_grease_corr_drop = corr_drop.loc['Hair_Grease'].drop('Hair_Grease')

        # 3️⃣ Mean imputation
        df_mean = df_luke_raw.copy()
        df_mean['Hair_Grease'] = df_mean['Hair_Grease'].fillna(df_mean['Hair_Grease'].mean())
        corr_mean = df_mean[numeric_cols].corr()
        hair_grease_corr_mean = corr_mean.loc['Hair_Grease'].drop('Hair_Grease')

        # ---- Plotting ----
        st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:12px; border-radius:12px; text-align:center; margin-top:20px;'>"
        f"<h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Hair_Grease Correlation Comparison</h3>"
        f"</div>",
        unsafe_allow_html=True
        )
    

        cols = st.columns(3)

        with cols[0]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; font-size:20px; color:#2C3E50;'><b>Original</b></p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(hair_grease_corr_original.to_frame().T, annot=True, cmap="YlGn", cbar=False, ax=ax)
            ax.set_yticklabels(['Hair_Grease'], rotation=0)
            st.pyplot(fig)
            plt.close(fig)

        with cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; font-size:20px; color:#2C3E50;'><b>Dropped Nulls</b></p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(hair_grease_corr_drop.to_frame().T, annot=True, cmap="YlGn", cbar=False, ax=ax)
            ax.set_yticklabels(['Hair_Grease'], rotation=0)
            st.pyplot(fig)
            plt.close(fig)

        with cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; font-size:20px; color:#2C3E50;'><b>Mean Imputed</b></p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(hair_grease_corr_mean.to_frame().T, annot=True, cmap="YlGn", cbar=False, ax=ax)
            ax.set_yticklabels(['Hair_Grease'], rotation=0)
            st.pyplot(fig)
            plt.close(fig)
            
        st.markdown(f"""
<div style='background-color:#aececb; padding:14px; border-radius:12px;'>
    <p style='font-size:20px; color:{TEXT}; margin:0;'>
        <b>Observation:</b> The correlation values in both approaches—dropping missing values and mean imputation—remain virtually identical to the original dataset. 
        This is because the number of missing values in <b>Hair_Grease</b> is very low. Based on this, I chose <b>Approach 1</b> and simply dropped the missing entries.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)








