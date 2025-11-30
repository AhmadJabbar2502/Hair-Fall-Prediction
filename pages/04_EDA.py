import streamlit as st
import pandas as pd
from EDA import render_predict_page, render_luke_page, render_nutrition_page

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

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
CARD_COLOR ="#d4e6e4"
CARD_COLOR2 = "#dcf4e0"

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
    <h1 style='color:#5e928a; font-size:32px; margin-bottom:5px;'>Exploratory Data Analysis</h1>
    <p style='color:{HEADER_COLOR}; font-size:20px; margin-top:0px;'>
        This section explores the behavioral patterns in the dataset — uncovering how lifestyle, biological, and environmental factors relate to hair loss. The goal is to identify visible trends and possible predictors through visual analysis.
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

df_predict_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df_luke_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")
df_nutrition_cleaned = load_csv("Data/Cleaned_Nutrition_Dataset.csv")

# ======== DATASET SELECTOR ==========
dataset_option = st.selectbox(
    "Select Dataset:",
    ["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset", "Nutrition Dataset"],
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


# ---- Hair Health Prediction Dataset Bar Plot ----
if dataset_option == "Hair Health Prediction Dataset":
    render_predict_page(df_predict_cleaned)

elif dataset_option == "Luke Hair Loss Dataset":
    render_luke_page(df_luke_cleaned)
elif dataset_option == "Nutrition Dataset":
    render_nutrition_page(df_nutrition_cleaned)