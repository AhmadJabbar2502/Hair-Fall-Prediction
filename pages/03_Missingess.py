import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Missingness Analysis", layout="wide")

# ======== GLOBAL STYLES (match your other pages) ==========
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
    section[data-testid="stSidebar"] * {{ color: {SIDBAR_TEXT} !important; font-size: 22px !important; font-family: 'Helvetica Neue', sans-serif; }}
    html, body, [class*="css"] {{ font-size: 22px !important; color: {TEXT} !important; }}
    div[data-testid="stMetricLabel"] {{ font-size: 22px !important; color: #333 !important; }}
    div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}
    /* Small adjustments for plot captions and spacing */
    .missing-caption {{ font-size:16px; color:{TEXT}; }}
    </style>
""", unsafe_allow_html=True)


# ======== HEADER (same style) ==========
st.markdown(
    f"""
    <div style='background-color:{BASE_BG}; padding:12px; border-radius:40px;'>
        <h1 style='text-align:center; font-size:30px; color:#5e928a; margin:6px 0;'>Missingness Analysis</h1>
        <p style='text-align:center; font-size:20px; color: #2a5a55; margin:0px 0 15px 0;'>
            Visualize and investigate patterns of missing data across datasets.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
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
df_luke_cleaned = load_csv("Data/luke_hair_loss_documentation Cleaned.csv")

# ======== DATASET SELECTOR (match other pages) ==========
dataset_option = st.selectbox(
    "Select Dataset:",
    ["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset"],
    index=0
)

if dataset_option == "None":
    st.info("Select a dataset from the dropdown above to begin.")
else:
    # ===== Dataset banner =====
    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:32px; margin:6px 0 6px 0;'>{dataset_option}</h2></div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # Hair Prediction Dataset view
    # ----------------------------
    if dataset_option == "Hair Health Prediction Dataset":
        if df_predict_raw is None:
            st.error("Predict Hair Fall raw data not found.")
        else:
            hair_data_raw = df_predict_raw.copy()

            # Subset for the heatmap features (as you requested)
            subset_cols = ["Medical_Conditions", "Medications_and_Treatments", "Nutritional_Deficiencies"]
            # Make sure columns exist
            subset_cols = [c for c in subset_cols if c in hair_data_raw.columns]
            if not subset_cols:
                st.error("Required columns for missingness heatmap not found in raw data.")
            else:
                df_hair_data_subset = hair_data_raw[subset_cols]
                nan_array = df_hair_data_subset.isna().astype(int).to_numpy()

                # Layout: heatmap (wide) + missing counts table (narrow)
                fig, ax = plt.subplots(figsize=(11, 5))
                # greenish palette (YlGn) for the missingness visualization
                im = ax.imshow(nan_array.T, interpolation="nearest", aspect="auto", cmap="YlGn")
                ax.set_xlabel("Index", fontsize=13, color=TEXT)
                ax.set_ylabel("Features", fontsize=13, color=TEXT)
                ax.set_title("Missing Values Heatmap", fontsize=16, fontweight="600", color=HEADER_COLOR)
                ax.set_yticks(range(len(df_hair_data_subset.columns)))
                ax.set_yticklabels(df_hair_data_subset.columns, fontsize=12, color=TEXT)
                ax.grid(True, axis="y", linestyle="--", alpha=0.6)
                # colorbar to the right
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
                cbar.ax.tick_params(labelsize=10)
                st.pyplot(fig)
                plt.close(fig)


                st.markdown(f"<p style='font-size:18px; color:{TEXT};'><b>Missing Value Count (raw)</b></p>", unsafe_allow_html=True)
                missing_counts = hair_data_raw.isna().sum().reset_index().rename(columns={"index":"Feature", 0:"Missing Count"})
                st.dataframe(missing_counts, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ----------------------------
                # Interpretation card (same style)
                # ----------------------------
                st.markdown(
                    f"""
                    <div style='background-color:#E6F4EA; padding:14px; border-radius:10px;'>
                        <p style='font-size:20px; color:{TEXT}; margin:0;'>
                        <b>Observation:</b> At first glance the missingness appears similar to MCAR.
                        From the heatmap we see possible subtle patterns — we will run statistical tests
                        (e.g. Chi-squared) to check whether missingness relates to <b>Stress Level</b> or <b>Age Range</b>.
                        Early checks suggest that people with higher stress levels and higher age ranges
                        may be less likely to report Medical Conditions.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # ----------------------------
                # Create indicators & grouped heatmaps
                # ----------------------------
                # Map stress to numeric if not already encoded
                if "Stress" in hair_data_raw.columns and "Stress_Level" not in hair_data_raw.columns:
                    hair_data_raw['Stress_Level'] = hair_data_raw['Stress']  # keep original strings if present
                # Ensure Age_Range exists (use cleaned mapping if not)
                if "Age" in hair_data_raw.columns and "Age_Range" not in hair_data_raw.columns:
                    hair_data_raw['Age_Range'] = pd.cut(hair_data_raw['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)

                hair_data_raw["Medical_Conditions_missing"] = hair_data_raw["Medical_Conditions"].isna().astype(int)

                st.markdown(f"<p style='font-size:20px; color:{TEXT};'><b>Missingness vs Stress Level & Age Range</b></p>", unsafe_allow_html=True)

                col3, col4 = st.columns(2, gap="large")

                # Heatmap by Stress Level (green palette)
                with col3:
                    # sort by Stress_Level if exists, else use original
                    if "Stress_Level" in hair_data_raw.columns:
                        stress_sorted = hair_data_raw.sort_values("Stress_Level", na_position='last')
                    else:
                        stress_sorted = hair_data_raw.copy()
                    missing_stress_matrix = stress_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)

                    fig, ax = plt.subplots(figsize=(10, 2))
                    sns.heatmap(missing_stress_matrix, cmap="YlGn", cbar=True, ax=ax)
                    # xticks: compute centers for each unique stress label
                    unique_stress = list(stress_sorted["Stress_Level"].dropna().unique())
                    if len(unique_stress) > 0:
                        xticks = [np.mean(np.where(stress_sorted["Stress_Level"] == lvl)) for lvl in sorted(unique_stress)]
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(sorted(unique_stress), fontsize=11, rotation=0)
                    ax.set_yticks([0])
                    ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
                    ax.set_title("Missingness in Medical Conditions (by Stress Level)", fontsize=14, color=HEADER_COLOR)
                    st.pyplot(fig)
                    plt.close(fig)

                # Heatmap by Age Range (green palette)
                with col4:
                    if "Age_Range" in hair_data_raw.columns:
                        age_sorted = hair_data_raw.sort_values("Age_Range", na_position='last')
                        missing_age_matrix = age_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        sns.heatmap(missing_age_matrix, cmap="YlGn", cbar=True, ax=ax)
                        uniq_age = list(age_sorted["Age_Range"].dropna().unique())
                        if len(uniq_age) > 0:
                            xticks = [np.mean(np.where(age_sorted["Age_Range"] == lvl)) for lvl in uniq_age]
                            ax.set_xticks(xticks)
                            ax.set_xticklabels(uniq_age, fontsize=11, rotation=0)
                        ax.set_yticks([0])
                        ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
                        ax.set_title("Missingness in Medical Conditions (by Age Range)", fontsize=14, color=HEADER_COLOR)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.markdown("<p style='font-size:16px; color:#666;'>Age_Range not available in raw data.</p>", unsafe_allow_html=True)

    # ----------------------------
    # Luke Hair Dataset view
    # ----------------------------
    else:  # Luke Hair Loss Dataset
        if df_luke_raw is None:
            st.error("Luke raw data not found.")
        else:
            df = df_luke_raw.copy()

            st.markdown(f"<p style='font-size:18px; color:{TEXT};'>The Luke dataset — visualize missingness below.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Full missingness heatmap (greenish)
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.heatmap(df.isna(), cmap="YlGn", cbar=False, ax=ax)
            ax.set_title("Missingness Heatmap — Luke Dataset", fontsize=16, color=HEADER_COLOR)
            st.pyplot(fig)
            plt.close(fig)

            # Missing counts table
            st.markdown(f"<p style='font-size:18px; color:{TEXT};'><b>Missing Value Count (Luke raw)</b></p>", unsafe_allow_html=True)
            st.dataframe(df.isna().sum().reset_index().rename(columns={"index":"Feature", 0:"Missing Count"}), use_container_width=True)

# ===== Next Steps (same footer style) =====
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:20px; color:{TEXT};'>"
    "Next, you can add interactive selectors to visualize missingness for specific features "
    "or to filter by date / stress / age ranges."
    "</p>",
    unsafe_allow_html=True
)
