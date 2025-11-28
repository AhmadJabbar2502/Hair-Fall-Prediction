# /About_the_Datasets/Nutrition_Dataset.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
BOXCOLOR = "#5d9189"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
SUB_HEADING_BG = "#749683"
SECONDARY = "#E67E22"

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
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
        colors = ["#85ada6", "#FFB347", "#D9D9D9", "#5ecbb9"]
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
        textprops={'fontsize':10, 'color':'#2C3E50'}
    )
    centre_circle = plt.Circle((0,0),0.50,fc="#FFFFFF")
    fig.gca().add_artist(centre_circle)
    ax.set_title(title, fontsize=16, color="#2C3E50")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# ---------------- MAIN FUNCTION ----------------
def show_nutrition_dataset(path):
    df = load_csv(path)
    df_cleaned = load_csv('Data/Cleaned_Nutrition_Dataset.csv')
    if df is None:
        return

    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:30px; margin:6px 0 6px 0;'>Nutrition Dataset Overview</h2></div>", unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px; color:#2C3E50; margin:4px 0 8px 0;'>The Comprehensive Nutritional Food Database provides detailed nutritional information (per 100 grams) for a wide range of food items commonly consumed around the world. This dataset aims to support dietary planning, nutritional analysis, and educational purposes by providing extensive data on the macro and micronutrient content of foods.</p>", unsafe_allow_html=True)

    n_rows, n_cols = df.shape
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Source:</b> Taken from Kaggle</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Rows:</b> {n_rows:,} &nbsp;&nbsp; <b>Features:</b> {n_cols}</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:22px; color:#2C3E50;'><b>Sample rows (raw, before cleaning)</b></p>", unsafe_allow_html=True)
    st.dataframe(df.head(8))
    
    
    # KPIs - example: unique foods, average calories, number of nutrients tracked
    c1, c2, c3 = st.columns(3)
    
    num_unique_foods = df['food'].nunique() if 'food' in df.columns else "N/A"
    avg_calories = df['Caloric Value'].mean() if 'Caloric Value' in df.columns else "N/A"
    num_nutrients = 34
    
    card_html = lambda title, value: f"""
        <div style='background:#5d9189; border-radius:12px; padding:12px; box-shadow:0 2px 6px rgba(0,0,0,0.06);'>
            <div style='font-size:18px; color:#FFFFFF; margin-bottom:6px;'>{title}</div>
            <div style='font-size:28px; color:#FFFFFF; font-weight:700;'>{value}</div>
        </div>
    """
    
    with c1: st.markdown(card_html("Unique Foods", num_unique_foods), unsafe_allow_html=True)
    with c2: st.markdown(card_html("Average Calories", f"{avg_calories:.1f}" if isinstance(avg_calories,float) else avg_calories), unsafe_allow_html=True)
    with c3: st.markdown(card_html("Tracked Nutrients", num_nutrients), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    nutrients = ['Caloric Value','Fat','Saturated Fats','Carbohydrates','Protein','Sugars','Dietary Fiber']
    
   # Donut Charts Section
    st.markdown(
        f"<h3 style='color:{TEXT}; text-align:center; font-size:25px;'>Nutritional Donut Visualizations</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)

    with col1:
        df['Macro_Dominant'] = df[['Protein','Fat','Carbohydrates']].idxmax(axis=1)
        img = create_donut_image(df, 'Macro_Dominant', 'Macro Dominance of Foods')
        st.image(img, width=370)

    with col2:
        def caloric_category(x):
            if x < 50: return "Very Low (<50)"
            elif x < 150: return "Low (50-150)"
            elif x < 300: return "Moderate (150-300)"
            else: return "High (>300)"

        df['Calorie_Category'] = df['Caloric Value'].apply(caloric_category)

        img_calorie = create_donut_image(df, 'Calorie_Category', 'Caloric Density Distribution')
        st.image(img_calorie, width=450)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    

    # Boxplots for key nutrients
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df[nutrients], palette=["#c1dab8", "#94b89e", "#679988"], ax=ax)
    ax.set_title("Boxplot of Key Nutrients", fontsize=12, color=TEXT)
    st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bar plot for top 5 calorie-rich foods
    top_calories = df[['food','Caloric Value']].nlargest(10,'Caloric Value')
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=top_calories, x='Caloric Value', palette=["#c1dab8"], y='food', ax=ax)
    ax.set_title("Top 5 Calorie-Rich Foods", fontsize=12, color=TEXT)
    st.pyplot(fig)
    
  
    
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # --- Download cleaned dataset ---
    st.markdown("<br>", unsafe_allow_html=True)

    # Heading
    st.markdown(
        f"""
        <div style='background-color:{SUB_HEADING_BG}; padding:10px; text-align:center; border-radius:10px;'>
            <h2 style='color:{ACCENT}; font-size:30px; margin:6px 0 6px 0;'>Download the Datasets</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    # Description text
    st.markdown(
        f"<p style='text-align:center; font-size:22px; color:{TEXT};'>"
        "Here you can download both the original (raw) dataset and the cleaned version used for analysis."
        "</p>",
        unsafe_allow_html=True
    )


    # Download buttons (vertical stack)
    with st.spinner("Preparing datasets for download..."):
        col = st.container()

        with col:
            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=df_cleaned.to_csv(index=False).encode('utf-8'),
                file_name="Predict_Hair_Fall_Cleaned.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.download_button(
                label="📄 Download Raw Dataset",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="Predict_Hair_Fall_Raw.csv",
                mime="text/csv",
                use_container_width=True
            )
            





