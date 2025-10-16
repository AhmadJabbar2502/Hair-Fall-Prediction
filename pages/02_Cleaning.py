import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Cleaning", layout="wide")

# ---- HEADER ----
st.markdown(
    "<h1 style='text-align:center; color:#2E8B57; font-size:36px;'>Cleaning the Data Sets</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:22px; color:#555;'>"
    "Before analysis, the datasets were cleaned and prepared, handling missing values, "
    "encoding categorical features, and creating meaningful derived variables."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

# ---- Load Datasets ----
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

df_raw = load_csv("Data/Predict Hair Fall Raw.csv")
df_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")

if df_raw is None or df_cleaned is None:
    st.error("One or more dataset files not found.")
else:
    # ---- Dataset Banner ----
    st.markdown(
        "<div style='background-color:#F0F0F0; padding:12px; border-radius:8px; text-align:center;'>"
        "<h2 style='color:#2E8B57; font-size:28px;'>Hair Health Prediction Dataset</h2></div>",
        unsafe_allow_html=True
    )

    # ---- Columns & Unique Values Table (raw dataset) ----
    st.markdown("<p style='font-size:22px; color:#555;'>Unique values in each column:</p>", unsafe_allow_html=True)
    unique_values_dict = {col: df_raw[col].dropna().unique() for col in df_raw.columns}
    unique_values_list = [(col, ', '.join(map(str, vals))) for col, vals in unique_values_dict.items()]
    unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
    st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))

    # ---- Encoded Variables ----
    st.markdown("<p style='font-size:22px; color:#555;'><b>Encoded Variables:</b></p>", unsafe_allow_html=True)
    encoded_vars_names = ['Genetics', 'Hormonal_Changes', 'Poor_Hair_Care_Habits', 
                          'Environmental_Factors', 'Smoking', 'Weight_Loss']
    for var in encoded_vars_names:
        st.markdown(f"- <b>{var}</b>", unsafe_allow_html=True)

    # ---- Derived Variables ----
    st.markdown("<p style='font-size:22px; color:#555;'><b>Derived Variables:</b></p>", unsafe_allow_html=True)
    st.markdown("- Stress_Level: Encoded ordinal variable (Low=0, Moderate=1, High=2)", unsafe_allow_html=True)
    st.markdown("- Age_Range: Binned age ranges (18-30, 30-40, 40-51)", unsafe_allow_html=True)

    # ---- Visualizations (cleaned dataset) ----
    st.markdown("<p style='font-size:22px; color:#555;'><b>Age Range Distribution</b></p>", unsafe_allow_html=True)
    df_cleaned['Age_Range'] = pd.cut(df_cleaned['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
    age_counts = df_cleaned['Age_Range'].value_counts().sort_index()
    st.bar_chart(age_counts)

    st.markdown("<p style='font-size:22px; color:#555;'><b>Stress Level Distribution</b></p>", unsafe_allow_html=True)
    df_cleaned['Stress_Level'] = df_cleaned['Stress'].map({'Low':0,'Moderate':1,'High':2})
    stress_counts = df_cleaned['Stress_Level'].value_counts().sort_index()
    st.bar_chart(stress_counts)

    # ---- Next Steps ----
    st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:20px; color:#555;'>"
        "Next, we will explore missing values and inconsistencies in the datasets "
        "before performing modeling and analysis."
        "</p>",
        unsafe_allow_html=True
    )
