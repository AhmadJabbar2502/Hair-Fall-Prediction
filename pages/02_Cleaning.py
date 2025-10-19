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
import os, sys, traceback

@st.cache_data
def load_csv(path):
    # Helpful diagnostics for deploy vs local
    st.write(f"Attempting to load: {path}")
    st.write("CWD:", os.getcwd())
    st.write("Files in cwd:", os.listdir(".")[:50])
    # also show contents of Data/ if it exists
    if os.path.isdir("Data"):
        st.write("Files in Data/:", os.listdir("Data")[:200])
    st.write("Python version:", sys.version)
    try:
        import pandas as pd
        st.write("pandas version:", pd.__version__)
    except Exception:
        st.write("Unable to import pandas for version check")

    # Try load with clear errors
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path} (CWD: {os.getcwd()})")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading CSV at {path}: {e}")
        st.exception(e)
        raise
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Loaded object is not DataFrame: {type(df)}")
    return df

df_predict_raw = load_csv("Data/Predict Hair Fall Raw.csv")
df_predict_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df_luke_raw = load_csv("Data/Luke_hair_loss_documentation Raw.csv")
df_luke_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

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

        # ===== Data Cleaning Summary =====
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:26px; color:{TEXT};'><b>Data Cleaning Summary — Hair Health Prediction Dataset</b></p>", unsafe_allow_html=True)

        # 1️⃣ ID Column Handling
        st.markdown(f"""
        <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <h3 style='color:{ACCENT}; font-size:32px; margin:6px 0;'>Unique Values After Cleaning</h3>
            <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
                The table below summarizes all unique entries in each column after applying the cleaning and encoding steps.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Unique values table ----
        unique_values_list = []

        for col in df_cleaned.columns:
            vals = df_cleaned[col].dropna().unique()
            
            # Sort numeric values, keep categorical as-is
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                vals = sorted(vals)
            else:
                vals = list(vals)
            
            unique_values_list.append((col, ', '.join(map(str, vals))))

        unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
        st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))


        # ---- Visualizations ----

        st.markdown(f"""
        <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:32px; margin:6px 0;'>Visualizations After Cleaning</h3>
            <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
                The following plots highlight the distribution of key variables — including Age Ranges and Stress Levels — 
                after performing data cleaning and encoding.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Encode and Prepare Columns for Plotting ----
        df['Age_Range'] = pd.cut(df['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
        df['Stress_Level'] = df['Stress'].map({'Low':0,'Moderate':1,'High':2})

        # ---- Plots ----
        fig, axes = plt.subplots(1,2, figsize=(14,4))  # Two plots: Age Range & Stress Level
        sns.countplot(x='Age_Range', data=df, palette=['#86aca9','#5d9189','#A8D5BA'], ax=axes[0])
        axes[0].set_title("Age Range Distribution", fontsize=16)
        sns.countplot(x='Stress_Level', data=df, palette=['#86aca9','#5d9189','#A8D5BA'], ax=axes[1])
        axes[1].set_title("Stress Level Distribution", fontsize=16)

        for ax in axes:
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        st.pyplot(fig)


    else:
        df = df_luke_cleaned
        
        # ===== Data Cleaning Summary =====
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:26px; color:{TEXT};'><b>Data Cleaning Summary — Luke Hair Loss Dataset</b></p>", unsafe_allow_html=True)

        # 1️⃣ Variable Renaming
        st.markdown(f"""
        <div style='background-color:{BASE_BG}; padding:10px 20px; border-left:6px solid #5d9189; border-radius:10px;'>
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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
            <p style='font-size:22px; color:{TEXT}; margin:0;'>
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

        # Continue to show the unique values table
        st.markdown(f"""
        <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:32px; margin:6px 0;'>Unique Values After Cleaning</h3>
            <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
                The table below summarizes all unique entries in each column after applying the cleaning and encoding steps.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---- Unique values table ----
        unique_values_list = []

        for col in df_luke_cleaned.columns:
            vals = df_luke_cleaned[col].dropna().unique()
            
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df_luke_cleaned[col]):
                vals = sorted(vals)  # Sort numeric values
            else:
                vals = list(vals)  # Keep categorical values as-is
            
            unique_values_list.append((col, ', '.join(map(str, vals))))

        unique_df = pd.DataFrame(unique_values_list, columns=["Column", "Values"])
        st.dataframe(unique_df.style.set_properties(**{'font-size':'20px'}))


        st.markdown(f"""
            <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
                <h3 style='color:{ACCENT}; font-size:32px; margin:6px 0;'>Visualizations After Cleaning</h3>
                <p style='color:{BASE_BG}; font-size:20px; margin:0;'>
                    The following plots highlight the distribution of key variables — including Hair Loss, Stress Level, and Pressure Level
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
    
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
