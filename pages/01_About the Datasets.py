import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Data — Datasets Overview", layout="wide")

# ---- GLOBAL STYLES (kept in main) ----
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
BG_COLOR = "#0C0505"
SIDBAR_TEXT ="#202424"

st.markdown("""
    <style>
    .stApp { background-color:#EFEFEF; }
    section[data-testid="stSidebar"] { background-color: #2a5a55; padding: 16px 12px; }
    section[data-testid="stSidebar"] * { color: #a9cac6 !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }
    html, body, [class*="css"] { font-size: 18px !important; color: #FFFFFF !important; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetricLabel"] { font-size: 18px !important; color: #333 !important; }
    div[data-testid="stDataFrame"] { background-color: white !important; border: 3px solid #2a5a55 !important; border-radius: 12px !important; box-shadow: none !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stDataFrame"] {
    background-color: white !important;
    border: 3px solid #2a5a55 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': BASE_BG,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'text.color': TEXT,
})

st.markdown(
    f"""
    <div style='background-color:{BASE_BG}; padding:12px; border-radius: 40px;'>
        <h1 style='text-align:center; font-size:30px; color:#5e928a; margin:6px 0;'>Datasets Overview</h1>
        <p style='text-align:center; font-size:20px; color: #2a5a55; margin:0px 0 15px 0;'>
            Using two datasets related to hair loss and baldness, we explore their features and structure.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Import the dataset-specific modules

from About_the_Datasets.Predict_Hair_Fall import show_predict_hair_fall
from About_the_Datasets.Luke_Hair_Loss import show_luke_hair_loss
from About_the_Datasets.Nutrition_Dataset import show_nutrition_dataset

st.markdown("<hr style='border:0px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

dataset_choice = st.selectbox(
    "Select dataset to view (or choose 'None')",
    options=["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset", "Nutrition Dataset"],
    index=0
)

if dataset_choice == "Hair Health Prediction Dataset":
    show_predict_hair_fall(
        raw_path="Data/Predict Hair Fall.csv",
        cleaned_path="Data/Predict Hair Fall Cleaned.csv"
    )
elif dataset_choice == "Luke Hair Loss Dataset":
    show_luke_hair_loss(
        raw_path="Data/Luke_hair_loss_documentation.csv",
        cleaned_path="Data/Luke_hair_loss_documentation Cleaned.csv"
    )
elif dataset_choice == "Nutrition Dataset":
    show_nutrition_dataset(
        path="Data/Nutrition_Dataset.csv"
    )
else:
    st.info("Select a dataset from the dropdown above to begin. (You may also leave it as 'None')")
