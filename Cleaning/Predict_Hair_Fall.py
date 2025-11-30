# Cleaning/Predict_Hair_Fall.py
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
    Render the Hair Health Prediction Dataset cleaning page.
    This contains the same text, unique-values table, and visualizations from your original file.
    """
    # Load dataset(s) locally within this module
    df_predict_raw = load_csv("Data/Predict Hair Fall Raw.csv")
    df_predict_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")

    # If cleaned file not present, fallback to raw for visual/unique display where possible
    df = df_predict_cleaned if df_predict_cleaned is not None else df_predict_raw

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px; color:{TEXT};'><b>Data Cleaning Summary — Hair Health Prediction Dataset</b></p>", unsafe_allow_html=True)

    # 1️⃣ ID Column Handling
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>ID Column Standardization</b><br>
            The original <code>ID</code> column was removed and replaced with a new identifier, 
            as the original only contained sequential assignment without analytical value.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2️⃣ Standardized Column Names
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Standardized Column Names</b><br>
            All variable names were standardized by capitalizing words and replacing spaces with underscores.<br>
            For example, <code>hormonal_changes</code> → <code>Hormonal_Changes</code>, 
            <code>medications_and_treatments</code> → <code>Medications_and_Treatments</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3️⃣ Binary Variable Encoding
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5e928a; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Binary Variable Encoding</b><br>
            Converted categorical (Yes/No) variables into numerical format for analysis:<br>
            <code>Genetics, Hormonal_Changes, Poor_Hair_Care_Habits, Environmental_Factors, Smoking, Weight_Loss</code>
            were encoded as <b>No = 0</b> and <b>Yes = 1</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4️⃣ Ordinal Variable Encoding
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #2a5a55; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Ordinal Variable Encoding</b><br>
            Assigned numerical values to represent increasing intensity levels in ordinal features:<br>
            - <code>Stress</code>: Low=0, Moderate=1, High=2
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5️⃣ Age Range Creation
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Derived Age Ranges</b><br>
            Created binned age ranges for better analysis:<br>
            - Bins: [18-30, 30-40, 40-51]<br>
            - Implementation: <code>pd.cut(Age, bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6️⃣ Trailing Spaces Removal
    st.markdown(f"""
    <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #E67E22; border-radius:10px;'>
        <p style='font-size:16px; color:{TEXT}; margin:0;'>
            <b>Cleaned Trailing Spaces</b><br>
            Removed trailing spaces from categorical variables to ensure consistency. Examples:<br>
            - <code>Medical_Conditions</code>: "Eczema " → "Eczema"<br>
            - <code>Medications_and_Treatments</code>: cleaned similarly<br>
            - <code>Nutritional_Deficiencies</code>: cleaned similarly
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== Heading Before Table =====
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Unique Values After Cleaning</h3>
        <p style='color:{BASE_BG}; font-size:18px; margin:0;'>
            The table below summarizes all unique entries in each column after applying the cleaning and encoding steps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # If no df available, show warning
    if df is None:
        st.warning("Cleaned or raw CSV not found at Data/Predict Hair Fall *.csv — please place CSV in Data/ and retry.")
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

    # ---- Visualizations ----
    st.markdown(f"""
    <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Visualizations After Cleaning</h3>
        <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
            The following plots highlight the distribution of key variables — including Age Ranges and Stress Levels — 
            after performing data cleaning and encoding.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prepare columns for plotting safely (check columns exist)
    if 'Age' in df.columns:
        try:
            df['Age_Range'] = pd.cut(df['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
        except Exception:
            df['Age_Range'] = None
    else:
        df['Age_Range'] = None

    if 'Stress' in df.columns:
        df['Stress_Level'] = df['Stress'].map({'Low':0,'Moderate':1,'High':2})
    else:
        df['Stress_Level'] = None

    # ---- Plots ----
    fig, axes = plt.subplots(1,2, figsize=(14,4))
    sns.countplot(x='Age_Range', data=df, palette=['#86aca9','#5d9189','#A8D5BA'], ax=axes[0])
    axes[0].set_title("Age Range Distribution", fontsize=16)
    sns.countplot(x='Stress_Level', data=df, palette=['#86aca9','#5d9189','#A8D5BA'], ax=axes[1])
    axes[1].set_title("Stress Level Distribution", fontsize=16)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    st.pyplot(fig)
