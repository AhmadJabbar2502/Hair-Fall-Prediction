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

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

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

def show_predict_hair_fall(raw_path, cleaned_path):
    
    df_raw = load_csv(raw_path)
    df_cleaned = load_csv(cleaned_path)

    if df_raw is None or df_cleaned is None:
        st.error("Dataset file not found.")
        return

    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:10px; text-align:center; border-radius:10px;'>"
        f"<h2 style='color:{ACCENT}; font-size:30px; margin:6px 0 6px 0;'>Hair Health Prediction Dataset</h2></div>", unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px; color:#2C3E50; margin:4px 0 8px 0;'>This dataset contains information about various factors that may contribute to baldness in individuals. Each row represents a unique individual, and the columns reflect genetics, hormonal changes, medical conditions, medications and treatments, nutritional deficiencies, stress levels, age, hair care habits, environmental factors, smoking, weight loss and presence/absence of baldness.</p>", unsafe_allow_html=True)

    n_rows, n_cols = df_raw.shape
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Source:</b> Taken from Kaggle</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px; color:#2C3E50;'><b>Rows:</b> {n_rows:,} &nbsp;&nbsp; <b>Features:</b> {n_cols}</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:22px; color:#2C3E50;'><b>Sample rows (raw, before cleaning)</b></p>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(8))

    # ---------- AGE SLIDER (moved here) ----------
    # Minimal, non-invasive age preview: slider -> count + head preview
    age_col = detect_target(["Age", "age"], df_cleaned)
    if age_col:
        st.markdown("<div style='margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:22px; color:{TEXT};'><b>Age preview:</b> Use the slider to filter and preview rows by age.</p>", unsafe_allow_html=True)
        # compute min/max safely 
        ages_series = pd.to_numeric(df_cleaned[age_col], errors="coerce").dropna()
        if not ages_series.empty:
            min_age = int(ages_series.min())
            max_age = int(ages_series.max())
            if min_age != max_age:
                age_range = st.slider("Select age range (preview only)", min_age, max_age, (min_age, max_age))
                preview_df = df_cleaned[
                    (pd.to_numeric(df_cleaned[age_col], errors="coerce") >= age_range[0]) &
                    (pd.to_numeric(df_cleaned[age_col], errors="coerce") <= age_range[1])
                ]
                st.markdown(f"<p style='font-size:16px; color:{TEXT};'><b>Rows in selected age range:</b> {len(preview_df):,}</p>", unsafe_allow_html=True)
                st.dataframe(preview_df.head(8))
            else:
                st.info(f"Age column present but all values are {min_age}.")
        else:
            st.info("Age column detected but contains no numeric values.")
        st.markdown("</div>", unsafe_allow_html=True)
    # ---------- end age slider ----------

    # detect columns
    target_col = detect_target(["Hair_Loss", "Hair Loss", "Baldness", "Bald", "hair_loss", "hair loss"], df_cleaned)

    # KPIs
    pct_hair_loss = None
    if target_col:
        vc = df_cleaned[target_col].value_counts(dropna=True)
        denom = vc.sum() if vc.sum() > 0 else np.nan
        pct_hair_loss = vc.get(1,0)/denom*100 if denom and not np.isnan(denom) else np.nan
    mean_age = None
    if age_col:
        ages = pd.to_numeric(df_cleaned[age_col], errors="coerce").dropna()
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

    # Age distribution
    if age_col:
        with st.spinner("Generating Age Distribution..."):
            fig, ax = plt.subplots(figsize=(13,4))
            ax.hist(pd.to_numeric(df_cleaned[age_col], errors="coerce").dropna(),
                    bins=15, color='#86aca9', alpha=0.8, edgecolor='white')
            ax.set_title("Age Distribution", fontsize=16, color=TEXT)
            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            st.pyplot(fig)

    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Categorical features with spinner
    st.markdown(
        f"<h3 style='color:{TEXT}; text-align:center; font-size:22px;'>Top Categorical Features</h3>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)

    with st.spinner("Generating categorical plots..."):
        if "Stress" in df_cleaned.columns:
            with c1:
                fig, ax = plt.subplots(figsize=(5,4))
                sns.countplot(data=df_cleaned, x="Stress", palette=["#c1dab8", "#94b89e", "#679988"], ax=ax)
                ax.set_title("Stress Levels", fontsize=14, color=TEXT)
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
                st.pyplot(fig)
                plt.close(fig)

        if "Genetics" in df_cleaned.columns:
            with c2:
                fig, ax = plt.subplots(figsize=(5,4))
                sns.countplot(data=df_cleaned, x="Genetics", palette=["#c1dab8", "#94b89e"], ax=ax)
                ax.set_title("Genetics", fontsize=14, color=TEXT)
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
                st.pyplot(fig)
                plt.close(fig)

        if "Poor_Hair_Care_Habits" in df_cleaned.columns:
            with c3:
                fig, ax = plt.subplots(figsize=(5,4))
                sns.countplot(data=df_cleaned, x="Poor_Hair_Care_Habits", palette=["#c1dab8", "#94b89e"], ax=ax)
                ax.set_title("Poor Hair Care Habits", fontsize=14, color=TEXT)
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Donut charts
    c1, c2 = st.columns(2)
    with st.spinner("Generating donut charts..."):
        with c1:
            med_img = create_donut_image(df_cleaned, "Medical_Conditions", "Top Medical Conditions (%)", top_n=5)
            if med_img:
                st.image(med_img)
        with c2:
            nut_img = create_donut_image(df_cleaned, "Nutritional_Deficiencies", "Top Nutritional Deficiencies (%)", top_n=5)
            if nut_img:
                st.image(nut_img)

    st.markdown("<hr style='border:1px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Hair loss distribution
    if "Hair_Loss" in df_cleaned.columns:
        with st.spinner("Generating Hair Loss distribution..."):
            st.markdown("<h3 style='color:#2C3E50; text-align:center; font-size:22px;'>Hair Loss Distribution</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(x='Hair_Loss', hue='Hair_Loss', data=df_cleaned,
                          palette=["#c1dab8", "#94b89e", "#2E8B57", "#2E8B57"], legend=False, ax=ax)
            ax.set_xlabel("Hair Loss", fontsize=12, color=TEXT)
            ax.set_ylabel("Count", fontsize=12, color=TEXT)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            st.pyplot(fig)
            plt.close(fig)
            
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
                data=df_raw.to_csv(index=False).encode('utf-8'),
                file_name="Predict_Hair_Fall_Raw.csv",
                mime="text/csv",
                use_container_width=True
            )
