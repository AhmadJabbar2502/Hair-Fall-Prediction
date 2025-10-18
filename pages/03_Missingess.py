import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

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

st.markdown(f"""
<style>
.stApp {{ background-color:#EFEFEF; }}
section[data-testid="stSidebar"] {{ background-color: {SECTION_BG}; padding: 16px 12px; }}
section[data-testid="stSidebar"] * {{ color: {SIDBAR_TEXT} !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }}
html, body, [class*="css"] {{ font-size: 22px !important; color: {TEXT} !important; }}
div[data-testid="stMetricLabel"] {{ font-size: 22px !important; color: #333 !important; }}
div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}
.missing-caption {{ font-size:16px; color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{BASE_BG}; padding:15px; border-radius:40px; text-align:center;'>
    <h1 style='color:#5e928a; font-size:32px; margin-bottom:5px;'>Missingness Analysis</h1>
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
        f"<h2 style='color:{ACCENT}; font-size:26px; margin:5px 0 5px 0;'>{dataset_option}</h2></div>",
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
                st.markdown(f"<p style='font-size:18px; color:{TEXT};'><b>Missing Value Counts</b></p>", unsafe_allow_html=True)
                missing_counts = hair_raw.isna().sum().reset_index().rename(columns={"index":"Feature", 0:"Missing Count"})
                st.dataframe(missing_counts, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:32px; margin:6px 0;'>Missingess in Medical Conditions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # ---- Observation Card ----
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
                <p style='font-size:20px; color:{TEXT}; margin:0;'>
                <b>Observation:</b> Missingness in <b>Medical_Conditions</b> is not completely random (not MCAR). Chi-squared tests reveal a strong dependency on <b>Stress Level</b> and <b>Age Range</b>. Higher stress and older age correlate with more missing entries.
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
            st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:23px; text-align:center'; >Chi-Squared Tests for Missingness</h2>", unsafe_allow_html=True)
            st.markdown(
                """
                <p style="font-size:20px; color:{TEXT}; line-height:1.5;">
                The missingness in <b>Medical Conditions</b> was tested against both <b>Stress Level</b> and <b>Age Range</b>.<br>
                The analysis confirms that missingness is <b>not random</b>.<br><br>
                - Probability that the difference between the observed and expected value of missingness due to Stress Level by chance: <b>7.756e-17</b><br>
                - Probability that the difference between the observed and expected value of missingness due to Age Range by chance: <b>2.335e-23</b><br><br>
                These extremely low probabilities indicate a strong relationship between missingness and these variables.
                </p>
                """.replace("{TEXT}", TEXT),
                unsafe_allow_html=True
            )


            # Contingency tables & bar plots
            contingency_stress = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Stress_Level'])
            contingency_age = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Age_Range'])
            chi2_stress, p_stress, dof_stress, ex_stress = stats.chi2_contingency(contingency_stress)
            chi2_age, p_age, dof_age, ex_age = stats.chi2_contingency(contingency_age)

            # Plot Observed vs Expected (Stress)
            fig, ax = plt.subplots(figsize=(15,6))
            obs = contingency_stress.loc[1]
            exp = pd.Series(ex_stress[1], index=contingency_stress.columns)
            x = np.arange(len(obs))
            width = 0.35
            ax.bar(x - width/2, obs, width, label='Observed', color='#86aca9')
            ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
            ax.set_xticks(x)
            ax.set_xticklabels(obs.index)
            ax.set_title("Missingness: Stress Level", fontsize=14, color=HEADER_COLOR)
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # Plot Observed vs Expected (Age)
            fig, ax = plt.subplots(figsize=(15,6))
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

            # ===== Imputation Info =====
            st.markdown(
                f"""
                <div style='background-color:#C7E9C0; padding:12px; border-radius:12px; text-align:center;'>
                    <h2 style='font-size:25px; color:#2E8B57; margin:0;'>Imputation of Missing Values</h2>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(f"""
            <p style='font-size:20px; color:{TEXT}; line-height:1.5;'>
            Missing <b>Medical_Conditions</b> were imputed using <b>Random Forest (n_estimators=1000)</b> 
            trained on features: Stress Level, Age Range, Genetic Encoding, Hormonal Changes, Nutritional Deficiencies, Smoking, Weight Loss, Environmental Factors.
            </p>
            """, unsafe_allow_html=True)

            # ---- Before vs After Imputation Plot ----
            df_raw = pd.read_csv("Data/Predict Hair Fall Raw.csv")
            df_cleaned = pd.read_csv("Data/Predict Hair Fall Cleaned.csv")

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
