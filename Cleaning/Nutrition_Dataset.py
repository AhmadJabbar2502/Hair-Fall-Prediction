# Cleaning/Nutrition_Dataset.py
import streamlit as st
import pandas as pd

# Duplicate color & style constants to keep page identical when module runs standalone
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
BG_COLOR = "#0C0505"
SIDBAR_TEXT = "#a9cac6"

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def run():
    df_nutrition_raw = load_csv("Data/Nutrition_Dataset.csv")
    df_nutrition_cleaned = load_csv("Data/Cleaned_Nutrition_Dataset.csv")

    df = df_nutrition_cleaned if df_nutrition_cleaned is not None else df_nutrition_raw

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Data Cleaning Summary — Nutrition Dataset</b></p>", unsafe_allow_html=True)

    # 1) Renamed column food to Food
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Renamed column</b><br>
            The column <code>food</code> was renamed to <code>Food</code> to standardize capitalization and align with other variable names.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 2) Dropped unnamed columns
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Dropped index/placeholder columns</b><br>
            Columns <code>Unnamed: 0</code> and <code>Unnamed: 0.1</code> were removed as they contained no analytical information (likely index remnants from file exports).
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 3) Standardized Column Names
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5e928a; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Standardized Column Names</b><br>
            All variable names were standardized by capitalizing words and replacing spaces with underscores.<br>
            For example: <code>Vitamin K</code> → <code>Vitamin_K</code>, <code>Nutrition Density</code> → <code>Nutrition_Density</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Unique Values heading
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Unique Values After Cleaning</h3>
        <p style='color:{BASE_BG}; font-size:18px; margin:0;'>
            The table below summarizes all unique entries in each column after applying the cleaning and encoding steps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.warning("Cleaned or raw CSV not found at Data/Nutrition_Dataset *.csv — please place CSV in Data/ and retry.")
        return

    # ---- Unique values table ----
    unique_values_list = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                vals = sorted(vals)
            except Exception:
                vals = list(vals)
        else:
            vals = list(vals)
        unique_values_list.append((col, ', '.join(map(str, vals))))

    unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
    st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))

    st.markdown("<br>", unsafe_allow_html=True)

