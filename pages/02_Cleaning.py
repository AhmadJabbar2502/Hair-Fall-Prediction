import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Data Cleaning", layout="wide")

# ======== GLOBAL STYLES ==========
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
    .stApp { background-color:#EFEFEF; }
    section[data-testid="stSidebar"] { background-color: #2a5a55; padding: 16px 12px; }
    section[data-testid="stSidebar"] * { color: #a9cac6 !important; font-size: 22px !important; font-family: 'Helvetica Neue', sans-serif; }
    html, body, [class*="css"] { font-size: 22px !important; color: #2C3E50 !important; }
    div[data-testid="stMetricLabel"] { font-size: 22px !important; color: #333 !important; }
    div[data-testid="stDataFrame"] { background-color: white !important; border: 3px solid #2a5a55 !important; border-radius: 12px !important; box-shadow: none !important; }
    </style>
""", unsafe_allow_html=True)

# ======== SIDEBAR ==========
st.sidebar.markdown(
    f"""
    <div style='padding:14px; border-radius:8px; background-color: {BASE_BG};'>
        <h3 style='margin:0; font-size:22px; color:{ACCENT};'>Hairfall Dashboard</h3>
        <p style='margin:6px 0 0 0; font-size:22px; color:{TEXT};'>
            Explore dataset cleaning and preprocessing.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ======== HEADER ==========
st.markdown(
    f"""
    <div style='background-color:{BASE_BG}; padding:12px; border-radius:40px;'>
        <h1 style='text-align:center; font-size:40px; color:#5e928a; margin:6px 0;'>Cleaning the Data Sets</h1>
        <p style='text-align:center; font-size:22px; color: #2a5a55; margin:0px 0 15px 0;'>
            Handling missing values, encoding categorical features, and creating derived variables before analysis.
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

# ======== DATASET SELECTOR ==========
dataset_choice = st.selectbox(
    "Select dataset to view:",
    options=["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset"],
    index=0
)

if dataset_choice == "None":
    st.info("Select a dataset from the dropdown above to begin.")
else:
    # ===== Dataset Section Banner =====
    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:35px; margin:6px 0 6px 0;'>{dataset_choice}</h2></div>",
        unsafe_allow_html=True
    )

    if dataset_choice=="Hair Health Prediction Dataset":
        df_raw = df_predict_raw
        df_cleaned = df_predict_cleaned
        df = df_cleaned

        # ---- Unique values table ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'>Unique values in each column:</p>", unsafe_allow_html=True)
        unique_values_dict = {col: df_raw[col].dropna().unique() for col in df_raw.columns}
        unique_values_list = [(col, ', '.join(map(str, vals))) for col, vals in unique_values_dict.items()]
        unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
        st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))

        # ---- Encoded variables ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Encoded Variables:</b></p>", unsafe_allow_html=True)
        encoded_vars_names = ['Genetics', 'Hormonal_Changes', 'Poor_Hair_Care_Habits', 
                              'Environmental_Factors', 'Smoking', 'Weight_Loss']
        for var in encoded_vars_names:
            st.markdown(f"- <b>{var}</b>", unsafe_allow_html=True)

        # ---- Derived variables ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Derived Variables:</b></p>", unsafe_allow_html=True)
        st.markdown("- Stress_Level: Encoded ordinal variable (Low=0, Moderate=1, High=2)", unsafe_allow_html=True)
        st.markdown("- Age_Range: Binned age ranges (18-30, 30-40, 40-51)", unsafe_allow_html=True)

        # ---- Visualizations ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Age Range Distribution</b></p>", unsafe_allow_html=True)
        df['Age_Range'] = pd.cut(df['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
        st.bar_chart(df['Age_Range'].value_counts().sort_index(), use_container_width=True)

        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Stress Level Distribution</b></p>", unsafe_allow_html=True)
        df['Stress_Level'] = df['Stress'].map({'Low':0,'Moderate':1,'High':2})
        st.bar_chart(df['Stress_Level'].value_counts().sort_index(), use_container_width=True)

    else:
        df = df_luke_cleaned

        # ---- Unique values table ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'>Unique values in each column:</p>", unsafe_allow_html=True)
        unique_values_dict = {col: df_luke_raw[col].dropna().unique() for col in df_luke_raw.columns}
        unique_values_list = [(col, ', '.join(map(str, vals))) for col, vals in unique_values_dict.items()]
        unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
        st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))

        # ---- Encoded variables 0/1 ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Encoded Variables (0/1):</b></p>", unsafe_allow_html=True)
        binary_vars = ['Swimming', 'Hair_Washing', 'Dandruff']
        for var in binary_vars:
            st.markdown(f"- <b>{var}</b>", unsafe_allow_html=True)
            df[var+"_Encoding"] = df[var].map({'No':0,'Yes':1})

        # ---- Encoded ordinal variables ----
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Encoded Variables (Ordinal):</b></p>", unsafe_allow_html=True)
        st.markdown("- Hair_Loss: No=0, Low=1, Moderate=2, High=3", unsafe_allow_html=True)
        st.markdown("- Stress_Level: Low=0, Moderate=1, High=2", unsafe_allow_html=True)
        st.markdown("- Pressure_Level: Low=0, Moderate=1, High=2", unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Distribution of Hair Loss, Stress, and Pressure Levels</b></p>", unsafe_allow_html=True)

        # ---- Plots ----
        fig, axes = plt.subplots(1,3, figsize=(18,4))
        sns.countplot(x='Hair_Loss', data=df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[0])
        axes[0].set_title("Hair Loss", fontsize=16)
        sns.countplot(x='Stress_Level', data=df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[1])
        axes[1].set_title("Stress Level", fontsize=16)
        sns.countplot(x='Pressure_Level', data=df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[2])
        axes[2].set_title("Pressure Level", fontsize=16)
        for ax in axes:
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
        st.pyplot(fig)
        
        # st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Age Range Distribution</b></p>", unsafe_allow_html=True)
        # st.bar_chart(df['Hair_Loss_Encoding'].value_counts().sort_index(), use_container_width=True)


# ===== Next Steps =====
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:20px; color:{TEXT};'>"
    "Next, we will explore missing values and inconsistencies in the datasets "
    "before performing modeling and analysis."
    "</p>",
    unsafe_allow_html=True
)
