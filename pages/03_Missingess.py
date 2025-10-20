import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from src.missingness import impute_medical_condition_mode
from src.medical_missingness import render_medical_missingness
from src.nutritional_missingness import render_nutritional_missingness


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
                
            df_raw = pd.read_csv("Data/Predict Hair Fall Raw.csv")
            df_cleaned = pd.read_csv("Data/Predict Hair Fall Cleaned.csv")
            render_medical_missingness(hair_raw=hair_raw, df_raw=df_raw, df_cleaned=df_cleaned,
                           HEADER_COLOR=HEADER_COLOR, TEXT=TEXT, SECTION_BG=SECTION_BG, ACCENT=ACCENT)
            
            render_nutritional_missingness(hair_raw=hair_raw, df_raw=df_raw, df_cleaned=df_cleaned,
                               HEADER_COLOR=HEADER_COLOR, TEXT=TEXT, SECTION_BG=SECTION_BG, ACCENT=ACCENT)

            
            st.markdown(f"""
                <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                    <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingness in Medications and Treatment</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
                <p style='font-size:16px; color:{TEXT}; margin:0;'>
                <b>Action taken:</b> I dropped the <b>2</b> missing records in <b>Medications and Treatment</b> from the dataset. 
                Removing these two rows has <b>no material effect</b> on the overall analyses or distributions given their extremely small count. 
                I verified they are not concentrated in a single subgroup (age/stress level).
            </div>
        """, unsafe_allow_html=True)


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








