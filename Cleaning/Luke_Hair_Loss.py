# Cleaning/Luke_Hair_Loss.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Duplicate color & style constants to keep page identical when module runs standalone
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
BG_COLOR = "#0C0505"
SIDBAR_TEXT ="#a9cac6"

# Dataset loader function (duplicated here per your instructions)
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def run():
    """
    Render the Luke Hair Loss Dataset cleaning page.
    Contains cleaning summary, unique values table, and visualizations from your original file.
    """
    df_luke_raw = load_csv("Data/Luke_hair_loss_documentation Raw.csv")
    df_luke_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

    df = df_luke_cleaned if df_luke_cleaned is not None else df_luke_raw

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Data Cleaning Summary — Luke Hair Loss Dataset</b></p>", unsafe_allow_html=True)

    # 1️⃣ Variable Renaming
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Standardized Column Names</b><br>
            All variable names were standardized by capitalizing words and replacing spaces with underscores. 
            <br>For instance, <code>hair loss</code> → <code>Hair_Loss</code> and <code>pressure level</code> → <code>Pressure_Level</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 2️⃣ Value Standardization
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Standardized Categorical Values</b><br>
            Modified categorical entries to ensure consistency across the dataset.<br>
            - <code>Hair_Washing</code>: replaced <b>Y</b> and <b>N</b> with <b>Yes</b> and <b>No</b>.<br>
            - <code>School_Assessment</code>: standardized to <b>Individual Assessment</b> and <b>Team Assessment</b> instead of abbreviations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 3️⃣ Binary Encoding
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5e928a; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Binary Variable Encoding</b><br>
            Converted categorical (Yes/No) variables into numerical format for analysis:<br>
            - <code>Swimming</code> and <code>Hair_Washing</code> were encoded as <b>No = 0</b> and <b>Yes = 1</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 4️⃣ Ordinal Encoding
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #2a5a55; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Ordinal Variable Encoding</b><br>
            Assigned numerical values to represent increasing intensity levels in ordinal features:<br>
            - <code>Hair_Loss</code>: No=0, Low=1, Moderate=2, High=3<br>
            - <code>Stress_Level</code>: Low=0, Moderate=1, High=2<br>
            - <code>Pressure_Level</code>: Low=0, Moderate=1, High=2<br>
            - <code>Dandruff</code>: None=0, Few=1, Many=2
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # If no df available, show warning
    if df is None:
        st.warning("Cleaned or raw CSV not found at Data/Luke_hair_loss_documentation *.csv — please place CSV in Data/ and retry.")
        return

    # Unique values table
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Unique Values After Cleaning</h3>
        <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
            The table below summarizes all unique entries in each column after applying the cleaning and encoding steps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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

    # Visualization header
    st.markdown(f"""
        <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Visualizations After Cleaning</h3>
            <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
                The following plots highlight the distribution of key variables — including Hair Loss, Stress Level, and Pressure Level
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Plots ----
    # Ensure required columns exist; otherwise fill with placeholder series to avoid crash
    plot_df = df.copy()
    for col in ['Hair_Loss', 'Stress_Level', 'Pressure_Level']:
        if col not in plot_df.columns:
            plot_df[col] = None

    fig, axes = plt.subplots(1,3, figsize=(18,4))
    sns.countplot(x='Hair_Loss', data=plot_df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[0])
    axes[0].set_title("Hair Loss", fontsize=16)
    sns.countplot(x='Stress_Level', data=plot_df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[1])
    axes[1].set_title("Stress Level", fontsize=16)
    sns.countplot(x='Pressure_Level', data=plot_df, palette=['#86aca9','#5d9189','#A8D5BA','#E67E22'], ax=axes[2])
    axes[2].set_title("Pressure Level", fontsize=16)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
    st.pyplot(fig)
