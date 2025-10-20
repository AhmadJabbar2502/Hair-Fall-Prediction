# app.py
import streamlit as st

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Hair Baldness Story", layout="centered")

# ======== COLORS ==========
BASE_BG = "#2a5a55"           # Main background
ACCENT = "#FFFFFF"            # Main text
SIDEBAR_BG = "#EFEFEF"        # Sidebar background
SIDEBAR_TEXT = "#000000"      # Sidebar text
TEXT = "#FFFFFF"              # Paragraph text
HEADER_COLOR = "#FFFFFF"      # Header color
BUTTON_BG = "#E67E22"         # Button background
BUTTON_HOVER = "#d35400"      # Button hover background

# ======== GLOBAL STYLES ==========
st.markdown(f"""
<style>
.stApp {{ background-color:{BASE_BG}; color:{TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; padding: 16px 12px; }}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_TEXT} !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }}
html, body, [class*="css"] {{ font-size:20px !important; color: {TEXT} !important; font-family: 'Helvetica Neue', sans-serif; }}
/* Button styles */
.next-btn {{
    background-color: {BUTTON_BG};
    color: #FFF;
    padding: 12px 25px;
    font-size: 18px;
    border-radius: 30px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
}}
.next-btn:hover {{
    background-color: {BUTTON_HOVER};
    transform: scale(1.05);
}}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{BASE_BG}; padding:20px; border-radius:40px; text-align:center;'>
    <h1 style='color:{ACCENT}; font-size:36px; margin-bottom:10px;'>Hair Baldness — A Data Story</h1>
    <p style='color:{ACCENT}; font-size:20px; margin-top:0px;'>
        Exploring lifestyle, biological, and environmental factors affecting hair loss through a data-driven narrative.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======== PROJECT DESCRIPTION ==========
st.markdown(f"""
<p style='font-size:18px; color:{TEXT}; margin-bottom:20px; line-height:1.6;'>
This project takes you on a data story journey from understanding dataset provenance, cleaning missing values, to uncovering insights through visualization. 
Two datasets are explored: the <b>Predict Hair Fall Dataset</b> and the <b>Luke Hair Loss Dataset</b>, capturing survey responses, lifestyle patterns, and daily logs related to hair health.
</p>
<p style='font-size:18px; color:{TEXT}; margin-bottom:20px; line-height:1.6;'>
The goal is to identify key predictors of hair loss — both genetic and environmental — and understand how multiple factors interact to influence hair health. 
By the end, users will gain insights into trends, correlations, and actionable patterns in the data.
</p>
""", unsafe_allow_html=True)

# ======== NEXT STEPS CARD ==========
st.markdown(f"""
<div style='background-color:{SIDEBAR_BG}; padding:15px 20px; border-left:6px solid {BUTTON_BG}; border-radius:10px; margin-top:20px;'>
    <h2 style='color:{SIDEBAR_TEXT}; margin-bottom:10px;'>Next Steps</h2>
    <p style='font-size:18px; color:{SIDEBAR_TEXT}; margin:0; line-height:1.5;'>
        Navigate through the sidebar to:
        <ul style='font-size:18px; color:{SIDEBAR_TEXT};'>
            <li>View raw and cleaned datasets and key summary statistics.</li>
            <li>Explore interactive visualizations showing feature distributions, relationships, and correlations.</li>
            <li>Discover insights about genetic, behavioral, and environmental factors affecting hair loss.</li>
        </ul>
    </p>
</div>
""", unsafe_allow_html=True)

