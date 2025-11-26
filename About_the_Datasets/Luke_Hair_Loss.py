import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
BOXCOLOR = "#5d9189"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
SECONDARY = "#E67E22"

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.markdown(FileNotFoundError)
        return None

def detect_target(col_candidates, df):
    for c in col_candidates:
        if c in df.columns:
            return c
    return None

def show_luke_hair_loss(raw_path, cleaned_path):
    df_raw = load_csv(raw_path)
    df_cleaned = load_csv(cleaned_path)

    if df_raw is None or df_cleaned is None:
        st.error("Dataset file not found.")
        return

    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:30px; margin:6px 0 6px 0;'>Luke Hair Loss Dataset</h2></div>", unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px; color:#2C3E50; margin:4px 0 8px 0;'>This dataset is about a postgraduate student who has been personally tracking hair loss issues since the age of 20. It records the student’s daily habits and various factors that could affect hair health over time. Hair loss was measured by placing one hand on the forehead, running the fingers through the hair toward the back of the head, and counting the number of hairs that fell on the hand. The data was collected over a period of 400 days.</p>", unsafe_allow_html=True)

    n_rows, n_cols = df_raw.shape
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Source:</b> Kaggle</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Rows:</b> {n_rows:,} &nbsp;&nbsp; <b>Features:</b> {n_cols}</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:22px; color:#2C3E50;'><b>Sample rows (raw, before cleaning)</b></p>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(8))

    # KPI cards
    c1, c2, c3 = st.columns(3)

    card_html = lambda title, value: f"""
        <div style='background:{BOXCOLOR}; border-radius:12px; padding:16px; 
                    box-shadow:0 2px 6px rgba(0,0,0,0.06); text-align:center;'>
            <div style='font-size:20px; color:#FFFFFF; margin-bottom:8px;'>{title}</div>
            <div style='font-size:30px; color:#FFFFFF; font-weight:700;'>{value}</div>
        </div>
    """

    with c1:
        st.markdown(card_html("Total Days Tracked", f"{len(df_cleaned):,}"), unsafe_allow_html=True)

    if "Coffee_Consumed" in df_cleaned.columns:
        avg_coffee = df_cleaned["Coffee_Consumed"].mean()
        with c2:
            st.markdown(card_html("Average Coffee Consumed", f"{avg_coffee:.1f}"), unsafe_allow_html=True)

    if "Brain_Working_Duration" in df_cleaned.columns:
        avg_brain = df_cleaned["Brain_Working_Duration"].mean()
        with c3:
            st.markdown(card_html("Average Brain Working Duration", f"{avg_brain:.1f}"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    numerical_cols = [c for c in ['Stay_Up_Late', 'Coffee_Consumed', 'Libido', 'Hair_Grease', 'Brain_Working_Duration'] if c in df_cleaned.columns]
    if numerical_cols:
        fig, ax = plt.subplots(figsize=(12,5))
        sns.boxplot(data=df_cleaned[numerical_cols], ax=ax, palette=[ACCENT, "#4C9F70", "#A8D5BA", SECONDARY, "#FFB347"])
        ax.set_xticklabels(numerical_cols, fontsize=12)
        ax.set_title("Boxplot of Daily Habits", fontsize=16, color=TEXT)
        st.pyplot(fig)

    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "Hair_Loss" in df_cleaned.columns:
        st.markdown("<h3 style='color:#2C3E50; text-align:center; font-size:22px;'>Hair Loss Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x='Hair_Loss', hue='Hair_Loss', data=df_cleaned, palette=["#c1dab8", "#94b89e", "#2E8B57", "#2E8B57"], legend=False, ax=ax)
        ax.set_xlabel("Hair Loss", fontsize=12, color=TEXT)
        ax.set_ylabel("Count", fontsize=12, color=TEXT)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        st.pyplot(fig)
        plt.close(fig)

    st.write("Next, we'll explore how missing data and inconsistencies were handled before analysis.")
