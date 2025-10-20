import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load datasets
df_predict_cleaned = pd.read_csv("Data/Predict Hair Fall Cleaned.csv")
df_luke_cleaned = pd.read_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

# Set page config
st.set_page_config(page_title="Hair Loss Insights Story", layout="wide")

# --------- Page Header ---------
st.title("Hair Loss Insights: Population & Personal Story")
st.markdown("""
Hair loss is influenced by genetics, nutrition, lifestyle, and environmental factors. 
This page explores population-level insights (1000 individuals) and individual-level patterns (Luke's daily data) to understand hair loss dynamics.
""")

# --------- Section 1: Genetics and Hair Loss ---------
st.header("Genetics and Hair Loss")

genetic_summary = df_predict_cleaned.groupby('Genetic_Encoding')['Hair_Loss'].value_counts(normalize=True).rename('proportion').reset_index()
fig_genetics = px.bar(
    genetic_summary, 
    x='Genetic_Encoding', y='proportion', color='Hair_Loss',
    labels={'Genetic_Encoding':'Genetic Predisposition', 'proportion':'Proportion', 'Hair_Loss':'Hair Loss'},
    color_discrete_map={0: 'lightblue', 1: 'orange'},
    text='proportion'
)
fig_genetics.update_layout(
    yaxis=dict(title='Proportion'),
    xaxis=dict(title='Genetic Predisposition'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=400
)
fig_genetics.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
st.plotly_chart(fig_genetics, use_container_width=True)

st.markdown("""
Genetics is a major predictor of hair loss:
- Individuals **with genetic predisposition**: 52% report hair loss
- Individuals **without genetic predisposition**: 48% report hair loss
""")

# --------- Section 2: Nutrition and Hair Health ---------
st.header("Nutrition and Hair Health")

# Example for Iron and Vitamin D deficiencies
nutrient_cols = ['Nutritional_Deficiencies', 'Hair_Loss', 'Genetic_Encoding']
df_nutrition = df_predict_cleaned[df_predict_cleaned['Genetic_Encoding']==1]  # Only genetically predisposed

# Count deficiencies
nutrient_summary = df_nutrition.groupby(['Hair_Loss', 'Nutritional_Deficiencies']).size().reset_index(name='count')
nutrient_summary['proportion'] = nutrient_summary.groupby('Hair_Loss')['count'].transform(lambda x: x/x.sum())

fig_nutrition = px.bar(
    nutrient_summary,
    x='Hair_Loss', y='proportion', color='Nutritional_Deficiencies',
    text='proportion',
    labels={'Hair_Loss':'Hair Loss', 'proportion':'Proportion', 'Nutritional_Deficiencies':'Nutrient Deficiency'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_nutrition.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', height=400
)
fig_nutrition.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
st.plotly_chart(fig_nutrition, use_container_width=True)

st.markdown("""
Among genetically predisposed individuals:
- **Iron deficiency**: 54% report hair loss
- **Vitamin D deficiency**: 58% report hair loss
Proper nutrition can help mitigate hair loss severity.
""")

# --------- Section 3: Lifestyle and Daily Factors (Luke) ---------
st.header("Lifestyle & Daily Factors: Luke's Hair Loss Data")

# Line chart: Hair Loss, Stress, Brain Working Hours
fig_luke = go.Figure()
fig_luke.add_trace(go.Scatter(
    x=df_luke_cleaned['Date'], y=df_luke_cleaned['Hair_Loss_Encoding'],
    mode='lines+markers', name='Hair Loss', line=dict(color='orange')
))
fig_luke.add_trace(go.Scatter(
    x=df_luke_cleaned['Date'], y=df_luke_cleaned['Stress_Level_Encoding'],
    mode='lines+markers', name='Stress Level', line=dict(color='purple')
))
fig_luke.add_trace(go.Scatter(
    x=df_luke_cleaned['Date'], y=df_luke_cleaned['Brain_Working_Duration'],
    mode='lines+markers', name='Brain Working Duration', line=dict(color='blue')
))
fig_luke.update_layout(
    xaxis_title='Date', yaxis_title='Encoding / Hours',
    plot_bgcolor='white', paper_bgcolor='white', height=450
)
fig_luke.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
st.plotly_chart(fig_luke, use_container_width=True)

st.markdown("""
Luke's daily data shows:
- Hair loss spikes coincide with **higher stress, longer brain-working hours, and late nights**
- Lifestyle changes can directly impact short-term hair shedding
""")

# --------- Section 4: Hair Grease vs Hair Loss ---------
st.header("Hair Grease and Hair Loss")

fig_grease = px.bar(
    df_luke_cleaned, x='Date', y=['Hair_Grease','Hair_Loss_Encoding'],
    barmode='group', labels={'value':'Level', 'variable':'Factor'},
    color_discrete_map={'Hair_Grease':'brown','Hair_Loss_Encoding':'orange'}
)
fig_grease.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', height=400
)
fig_grease.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
st.plotly_chart(fig_grease, use_container_width=True)

st.markdown("Higher hair grease buildup is associated with increased hair shedding.")

# --------- Section 5: Age & Alopecia Trends ---------
st.header("Age & Alopecia Trends (Population)")

alopecia_summary = df_predict_cleaned.groupby(['Age_Range','Hair_Loss']).size().reset_index(name='count')
fig_age = px.bar(
    alopecia_summary, x='Age_Range', y='count', color='Hair_Loss',
    color_discrete_map={0:'lightblue',1:'orange'},
    labels={'Age_Range':'Age Range', 'count':'Number of Individuals', 'Hair_Loss':'Hair Loss'}
)
fig_age.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', height=400
)
fig_age.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
st.plotly_chart(fig_age, use_container_width=True)

st.markdown("""
- Androgenetic Alopecia is most prevalent among individuals with a genetic predisposition.
- Ages 25–40 show the highest hair loss occurrences.
""")

# --------- Section 6: Recommendations ---------
st.header("Recommendations & Takeaways")

st.markdown("""
1. Maintain proper nutrition (iron, vitamin D).  
2. Manage stress and pressure levels.  
3. Optimize sleep and avoid late-night work.  
4. Moderate coffee intake and hair grease use.  
5. Early awareness for genetically predisposed individuals.
""")
