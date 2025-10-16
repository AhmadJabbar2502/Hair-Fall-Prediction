# app.py
import streamlit as st

st.set_page_config(page_title="Hair Baldness Story", layout="centered")

st.title("Hair Baldness — A Data Story")
st.write("""
This project explores datasets about factors that may contribute to baldness.
We tell the story from dataset provenance → cleaning & missingness → insights & models.
Use the **Data** page in the sidebar to see both datasets and quick key statistics.
""")
