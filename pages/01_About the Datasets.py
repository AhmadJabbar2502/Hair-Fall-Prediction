import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Data — Datasets Overview", layout="wide")
# ---- GLOBAL STYLES ----

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
    /* Change the main app background */
    .stApp {
        background-color:#EFEFEF; /* light gray background, change as you like */
    }


   /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #2a5a55;  /* sidebar background color */
        padding: 16px 12px;
    }

    /* --- SIDEBAR TEXT --- */
    section[data-testid="stSidebar"] * {
        color: #a9cac6 !important;   /* dark slate gray text */
        font-size: 22px !important;  /* smaller but readable font */
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* General text styling */
    html, body, [class*="css"] {
        font-size: 22px !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 22px !important;
        color: #333 !important;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---- DataFrame Table Styling ---- */

/* Target the dataframes inside Streamlit */
div[data-testid="stDataFrame"] {
    background-color: white !important;  /* Keep a clean white background */
    border: 3px solid #2a5a55 !important;  /* soft green-gray border */
    border-radius: 12px !important;
    box-shadow: none !important;  /* Remove any subtle shadows */
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

# ======== SIDEBAR ==========
st.sidebar.markdown(
    f"""
    <div style='padding:14px; border-radius:8px; background-color: #FFFFFF;'>
        <h3 style='margin:0; font-size:22px; color:{ACCENT};'>Hairfall Dashboard</h3>
        <p style='margin:6px 0 0 0; font-size:22px; color:{TEXT};'>
            Choose dataset and explore features.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ======== HEADER (page) ==========
st.markdown(
    f"""
    <div style='background-color:{BASE_BG}; padding:12px; border-radius: 40px;'>
        <h1 style='text-align:center; font-size:40px; color:#5e928a; margin:6px 0;'>Datasets Overview</h1>
        <p style='text-align:center; font-size:22px; color: #2a5a55; margin:0px 0 15px 0;'>
            Using two datasets related to hair loss and baldness, we explore their features and structure.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ======== DATA LOADING (same helper) ==========
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def detect_target(col_candidates, df):
    for c in col_candidates:
        if c in df.columns:
            return c
    return None

def create_donut_image(df, column, title, top_n=5, colors=None):
    if column not in df.columns:
        return None
    
    counts = df[column].dropna().astype(str).value_counts()
    if len(counts) > top_n:
        top_counts = counts[:top_n]
        other_count = counts[top_n:].sum()
        top_counts["Other"] = other_count
    else:
        top_counts = counts
    
    if colors is None:
        colors = ["#bddb8e", "#85ada6", "#A8D5BA", "#E67E22", "#FFB347", "#D9D9D9"]
        colors = colors[:len(top_counts)]
    
    explode = [0.05]*len(top_counts)
    fig, ax = plt.subplots(figsize=(5,5))
    wedges, texts, autotexts = ax.pie(
        top_counts, 
        labels=top_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75,
        colors=colors,
        explode=explode,
        textprops={'fontsize':10, 'color':TEXT}
    )
    centre_circle = plt.Circle((0,0),0.50,fc=BASE_BG)
    fig.gca().add_artist(centre_circle)
    ax.set_title(title, fontsize=16, color=TEXT)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# ======== FUNCTION TO RENDER DATASET SECTION (modified KPI cards and smaller fonts) ==========
def show_dataset_about(df_head, df_analysis, title, description="", source_text=None, show_head=True, plot_type='default'):
    if df_head is None or df_analysis is None:
        st.error("Dataset file not found.")
        return

    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:35px; margin:6px 0 6px 0;'>{title}</h2></div>", unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if description:
        st.markdown(f"<p style='font-size:22px; color:{TEXT}; margin:4px 0 8px 0;'>{description}</p>", unsafe_allow_html=True)

    n_rows, n_cols = df_head.shape
    if source_text:
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Source:</b> {source_text}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Rows:</b> {n_rows:,} &nbsp;&nbsp; <b>Features:</b> {n_cols}</p>", unsafe_allow_html=True)

    if show_head:
        st.markdown("<p style='font-size:30px; color:{TEXT};'><b>Sample rows (raw, before cleaning)</b></p>".format(TEXT=TEXT), unsafe_allow_html=True)
        st.dataframe(df_head.head(8))

    age_col = detect_target(["Age", "age"], df_analysis)
    target_col = detect_target(["Hair_Loss", "Hair Loss", "Baldness", "Bald", "hair_loss", "hair loss"], df_analysis)

    # ===== KPIs (rounded rectangular boxes) =====
    if plot_type=='default':
        pct_hair_loss = None
        if target_col:
            vc = df_analysis[target_col].value_counts(dropna=True)
            denom = vc.sum() if vc.sum() > 0 else np.nan
            pct_hair_loss = vc.get(1,0)/denom*100 if denom and not np.isnan(denom) else np.nan
        mean_age = None
        if age_col:
            ages = pd.to_numeric(df_analysis[age_col], errors="coerce").dropna()
            if len(ages)>0:
                mean_age = ages.mean()

        kpi1 = f"{pct_hair_loss:.2f}%" if pct_hair_loss and not np.isnan(pct_hair_loss) else "N/A"
        kpi2 = f"{mean_age:.1f}" if mean_age and not np.isnan(mean_age) else "N/A"

        c1, c2 = st.columns(2)

        card_html = lambda title, value: f"""
            <div style='background:{BOXCOLOR}; border-radius:12px; padding:12px; box-shadow:0 2px 6px rgba(0,0,0,0.06);'>
                <div style='font-size:22px; color:#FFFFFF; margin-bottom:6px;'>{title}</div>
                <div style='font-size:35px; color:#FFFFFF; font-weight:700;'>{value}</div>
            </div>
        """

        with c1:
            st.markdown(card_html("Percent Reporting Hair Loss", kpi1), unsafe_allow_html=True)
        with c2:
            st.markdown(card_html("Mean Age", kpi2), unsafe_allow_html=True)

        st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

    # ===== AGE DISTRIBUTION =====
    if age_col:
        fig, ax = plt.subplots(figsize=(13,4))
        ax.hist(pd.to_numeric(df_analysis[age_col], errors="coerce").dropna(),
                bins=15, color='#86aca9', alpha=0.8, edgecolor='white')
        ax.set_title("Age Distribution", fontsize=16, color=TEXT)
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)
        
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    
    # ===== CATEGORICAL FEATURES =====
    if title=="Hair Health Prediction Dataset":
        st.markdown(
            f"<h3 style='color:{TEXT}; text-align:center; font-size:26px;'>Top Categorical Features</h3>", 
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.countplot(data=df_analysis, x="Stress", palette=["#c1dab8", "#94b89e", "#679988"], ax=ax)
            ax.set_title("Stress Levels", fontsize=14, color=TEXT)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.countplot(data=df_analysis, x="Genetics", palette=["#c1dab8", "#94b89e"], ax=ax)
            ax.set_title("Genetics", fontsize=14, color=TEXT)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            st.pyplot(fig)
            plt.close(fig)

        with c3:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.countplot(data=df_analysis, x="Poor_Hair_Care_Habits", palette=["#c1dab8", "#94b89e"], ax=ax)
            ax.set_title("Poor Hair Care Habits", fontsize=14, color=TEXT)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

        # Donut charts
        c1, c2 = st.columns(2)
        with c1:
            med_img = create_donut_image(df_analysis, "Medical_Conditions", "Top Medical Conditions (%)", top_n=5)
            if med_img:
                st.image(med_img)
        with c2:
            nut_img = create_donut_image(df_analysis, "Nutritional_Deficiencies", "Top Nutritional Deficiencies (%)", top_n=5)
            if nut_img:
                st.image(nut_img)

    # ===== LUKE DATASET =====
    if title=="Luke Hair Loss Dataset":
        st.markdown("<div style='display:flex; gap:8px;'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        card_html = lambda title, value: f"""
            <div style='background:{BOXCOLOR}; border-radius:12px; padding:16px; 
                        box-shadow:0 2px 6px rgba(0,0,0,0.06); text-align:center;'>
                <div style='font-size:22px; color:#FFFFFF; margin-bottom:8px;'>{title}</div>
                <div style='font-size:36px; color:#FFFFFF; font-weight:700;'>{value}</div>
            </div>
        """

        with c1:
            st.markdown(card_html("Total Days Tracked", f"{len(df_analysis):,}"), unsafe_allow_html=True)

        if "Coffee_Consumed" in df_analysis.columns:
            avg_coffee = df_analysis["Coffee_Consumed"].mean()
            with c2:
                st.markdown(card_html("Average Coffee Consumed per Day", f"{avg_coffee:.1f}"), unsafe_allow_html=True)
       
        if "Brain_Working_Duration" in df_analysis.columns:
            avg_coffee = df_analysis["Brain_Working_Duration"].mean()
            with c2:
                st.markdown(card_html("Average Coffee Consumed per Day", f"{avg_coffee:.1f}"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        numerical_cols = ['Stay_Up_Late', 'Coffee_Consumed', 'Libido', 'Hair_Grease', 'Brain_Working_Duration']
        fig, ax = plt.subplots(figsize=(12,5))
        sns.boxplot(data=df_analysis[numerical_cols], ax=ax, palette=[ACCENT, "#4C9F70", "#A8D5BA", SECONDARY, "#FFB347"])
        ax.set_xticklabels(numerical_cols, fontsize=12)
        ax.set_title("Boxplot of Daily Habits", fontsize=16, color=TEXT)
        st.pyplot(fig)

    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "Hair_Loss" in df_analysis.columns:
        st.markdown("<h3 style='color:{0}; text-align:center; font-size:24px;'>Hair Loss Distribution</h3>".format(TEXT), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x='Hair_Loss', data=df_analysis, palette=["#c1dab8", "#94b89e", "#2E8B57"], ax=ax)
        ax.set_xlabel("Hair Loss", fontsize=12, color=TEXT)
        ax.set_ylabel("Count", fontsize=12, color=TEXT)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        st.pyplot(fig)
        plt.close(fig)

    st.write("Next, we'll explore how missing data and inconsistencies were handled before analysis.")

# ======== USAGE: dataset selection at start (single-select) ============
df_raw = load_csv("Data/Predict Hair Fall.csv")
df_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df2_raw = load_csv("Data/Luke_hair_loss_documentation.csv")
df2_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

st.markdown("<hr style='border:0px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

# dataset selector: default 'None'
dataset_choice = st.selectbox(
    "Select dataset to view (or choose 'None')",
    options=["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset"],
    index=0
)

if dataset_choice == "Hair Health Prediction Dataset":
    show_dataset_about(
        df_raw,
        df_cleaned,
        title="Hair Health Prediction Dataset",
        description="This dataset contains information about various factors that may contribute to baldness in individuals. Each row represents a unique individual, and the columns reflect genetics, hormonal changes, medical conditions, medications and treatments, nutritional deficiencies, stress levels, age, hair care habits, environmental factors, smoking, weight loss and presence/absence of baldness.",
        source_text="Taken from Kaggle",
        plot_type='default'
    )
elif dataset_choice == "Luke Hair Loss Dataset":
    show_dataset_about(
        df2_raw,
        df2_cleaned,
        title="Luke Hair Loss Dataset",
        description="This dataset is about a postgraduate student who has been personally tracking hair loss issues since the age of 20. It records the student’s daily habits and various factors that could affect hair health over time. Hair loss was measured by placing one hand on the forehead, running the fingers through the hair toward the back of the head, and counting the number of hairs that fell on the hand. The data was collected over a period of 400 days.",
        source_text="Kaggle",
        plot_type='boxplot'
    )
else:
    st.info("Select a dataset from the dropdown above to begin. (You may also leave it as 'None')")

