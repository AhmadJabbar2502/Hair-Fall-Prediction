import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

st.set_page_config(page_title="Data — Datasets Overview", layout="wide")

# ---- HEADER ----
st.markdown(
    "<h1 style='text-align:center; font-size:38px; color:#2F4F4F;'>Datasets Overview</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:22px; color:#4B4B4B;'>"
    "Using two datasets related to hair loss and baldness, we explore their features and structure."
    "</p>",
    unsafe_allow_html=True
)

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
        colors = plt.cm.Pastel1.colors[:len(top_counts)]
    
    explode = [0.05]*len(top_counts)
    
    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(
        top_counts, 
        labels=top_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75,
        colors=colors,
        explode=explode,
        textprops={'fontsize':9}
    )
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title(title, fontsize=18)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def show_dataset_about(df_head, df_analysis, title, description="", source_text=None, show_head=True, plot_type='default'):
    if df_head is None or df_analysis is None:
        st.error("Dataset file not found.")
        return

    st.markdown(
        f"<div style='background-color:#F0F0F0; padding:12px; text-align:center; border-radius:8px;'>"
        f"<h2 style='color:#2E8B57;'>{title}</h2></div>", unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if description:
        st.markdown(f"<p style='font-size:20px; color:#555;'>{description}</p>", unsafe_allow_html=True)

    n_rows, n_cols = df_head.shape
    if source_text:
        st.markdown(f"<p style='font-size:20px; color:#555;'><b>Source:</b> {source_text}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px; color:#555;'><b>Rows:</b> {n_rows:,} &nbsp;&nbsp; <b>Features:</b> {n_cols}</p>", unsafe_allow_html=True)

    if show_head:
        st.markdown("<p style='font-size:20px; color:#555;'><b>Sample rows (raw, before cleaning)</b></p>", unsafe_allow_html=True)
        st.dataframe(df_head.head(8))

    age_col = detect_target(["Age", "age"], df_analysis)
    target_col = detect_target(["Hair_Loss", "Hair Loss", "Baldness", "Bald", "hair_loss", "hair loss"], df_analysis)

    # KPIs
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

        kpi1 = f"{pct_hair_loss:.2f}%" if pct_hair_loss else "N/A"
        kpi2 = f"{mean_age:.1f}" if mean_age else "N/A"

        c1,c2 = st.columns(2)
        c1.metric("Percent Reporting Hair Loss", kpi1)
        c2.metric("Mean Age", kpi2)
        st.markdown("<hr style='border:1px solid #DDD;'>", unsafe_allow_html=True)

    # Age distribution
    if age_col:
        fig, ax = plt.subplots(figsize=(13,5))
        ax.hist(pd.to_numeric(df_analysis[age_col], errors="coerce").dropna(),
                bins=15, color="#2E8B57", alpha=0.7, edgecolor='white')
        ax.set_title("Age Distribution", fontsize=12)
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)
        
    st.markdown("<hr style='border:1px solid #AAA; margin:20px 0;'>", unsafe_allow_html=True)
    
    # Top categorical features for Hair Dataset (middle section)
    if title=="Hair Health Prediction Dataset":
        # Centered heading, bigger font, black color
        st.markdown(
            "<h3 style='color:black; text-align:center; font-size:32px;'>Top Categorical Features</h3>", 
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots(figsize=(6,6))
            sns.countplot(data=df_analysis, x="Stress", palette="pastel", ax=ax)
            ax.set_title("Stress Levels", fontsize=25)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(6,6))
            sns.countplot(data=df_analysis, x="Genetics", palette="pastel", ax=ax)
            ax.set_title("Genetics", fontsize=25)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            st.pyplot(fig)
            plt.close(fig)

        with c3:
            fig, ax = plt.subplots(figsize=(6,6))
            sns.countplot(data=df_analysis, x="Poor_Hair_Care_Habits", palette="pastel", ax=ax)
            ax.set_title("Poor Hair Care Habits", fontsize=25)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            st.pyplot(fig)
            plt.close(fig)



        st.markdown("<hr style='border:1px solid #AAA; margin:20px 0;'>", unsafe_allow_html=True)

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

    # Luke Dataset Boxplot + total days
    if title=="Luke Hair Loss Dataset":
        st.metric("Total Days Tracked", f"{len(df_analysis):,}")
        numerical_cols = ['stay_up_late', 'coffee_consumed', 'libido', 'hair_grease', 'brain_working_duration']
        fig, ax = plt.subplots(figsize=(12,6))
        sns.boxplot(data=df_analysis[numerical_cols], ax=ax)
        ax.set_xticklabels(numerical_cols, fontsize=14)
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Next, we'll explore how missing data and inconsistencies were handled before analysis.")

# --- Usage --- 
df_raw = load_csv("Data/Predict Hair Fall Raw.csv")
df_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df2 = load_csv("Data/Luke_hair_loss_documentation Raw.csv")

show_dataset_about(
    df_raw,
    df_cleaned,
    title="Hair Health Prediction Dataset",
    description="This dataset contains information about various factors that may contribute to baldness in individuals. Each row represents a unique individual, and the columns reflect genetics, hormonal changes, medical conditions, medications and treatments, nutritional deficiencies, stress levels, age, hair care habits, environmental factors, smoking, weight loss and presence/absence of baldness.",
    source_text="Taken from Kaggle",
    plot_type='default'
)

show_dataset_about(
    df2,  
    df2,
    title="Luke Hair Loss Dataset",
    description="This dataset is about a postgraduate student who has been personally tracking hair loss issues since the age of 20. It records the student’s daily habits and various factors that could affect hair health over time. Hair loss was measured by placing one hand on the forehead, running the fingers through the hair toward the back of the head, and counting the number of hairs that fell on the hand. The data was collected over a period of 400 days.",
    source_text="Kaggle",
    plot_type='boxplot'
)
