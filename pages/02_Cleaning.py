import streamlit as st
from Cleaning import Predict_Hair_Fall, Luke_Hair_Loss, Nutrition_Dataset

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Data Cleaning", layout="wide")

# ======== GLOBAL STYLES (KEEP UI CONSISTENT) ==========
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
BG_COLOR = "#0C0505"
SIDBAR_TEXT ="#a9cac6"

st.markdown("""
    <style>
    /* Main app background */
    .stApp { 
        background-color:#EFEFEF; 
    }

    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] { 
        background-color: #2a5a55; 
        padding: 16px 12px; 
    }

    /* --- SIDEBAR TEXT --- */
    section[data-testid="stSidebar"] * { 
        color: #a9cac6 !important;   
        font-size: 16px !important;
        font-family: 'Helvetica Neue', sans-serif; 
    }

    /* General text styling */
    html, body, [class*="css"] { 
        font-size:16px !important;
        color: #2C3E50 !important; 
        font-family: 'Helvetica Neue', sans-serif;
    }

    div[data-testid="stMetricLabel"] { 
        font-size: 18px !important; 
        color: #333 !important; 
    }

    /* DataFrame styling */
    div[data-testid="stDataFrame"] { 
        background-color: white !important; 
        border: 3px solid #2a5a55 !important; 
        border-radius: 12px !important; 
        box-shadow: none !important; 
    }
    </style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(
    f"""
    <div style='background-color:{BASE_BG}; padding:12px; border-radius:40px;'>
        <h1 style='text-align:center; font-size:25px; color:#5e928a; margin:6px 0;'>Cleaning the Data Sets</h1>
        <p style='text-align:center; font-size:20px; color: #2a5a55; margin:0px 0 15px 0;'>
            Handling missing values, encoding categorical features, and creating derived variables before analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

# ======== DATASET SELECTOR (DISPATCHER) ==========
dataset_choice = st.selectbox(
    "Select dataset to view (or choose 'None')",
    options=["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset", "Nutrition Dataset"],
    index=0
)


if dataset_choice == "None":
    st.info("Select a dataset from the dropdown above to begin.")
elif dataset_choice == "Hair Health Prediction Dataset":
    Predict_Hair_Fall.run()
elif dataset_choice == "Luke Hair Loss Dataset":
    Luke_Hair_Loss.run()
elif dataset_choice == "Nutrition Dataset":
    Nutrition_Dataset.run()

# ===== Next Steps (kept in main) =====
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:18px; color:{TEXT};'>"
    "Next, we will explore missing values and inconsistencies in the datasets "
    "before performing modeling and analysis."
    "</p>",
    unsafe_allow_html=True
)
