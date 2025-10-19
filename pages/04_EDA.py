import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import scipy.stats as stats
import numpy as np

# ======== PAGE CONFIG ==========
st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

# ======== GLOBAL STYLES ==========
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
BOXCOLOR = "#5d9189"
SECONDARY = "#E67E22"
TEXT = "#2C3E50"
SECTION_BG = "#2a5a55"
SECTION_BG_PLOTS = "#6f918d"
SIDBAR_TEXT = "#a9cac6"

st.markdown(f"""
<style>
.stApp {{ background-color:#EFEFEF; }}
section[data-testid="stSidebar"] {{ background-color: {SECTION_BG}; padding: 16px 12px; }}
section[data-testid="stSidebar"] * {{ color: {SIDBAR_TEXT} !important; font-size: 16px !important; font-family: 'Helvetica Neue', sans-serif; }}
html, body, [class*="css"] {{ font-size: 22px !important; color: {TEXT} !important; }}
div[data-testid="stMetricLabel"] {{ font-size: 22px !important; color: #333 !important; }}
div[data-testid="stDataFrame"] {{ background-color: white !important; border: 3px solid {SECTION_BG} !important; border-radius: 12px !important; box-shadow: none !important; }}
.missing-caption {{ font-size:16px; color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ==========
st.markdown(f"""
<div style='background-color:{BASE_BG}; padding:15px; border-radius:40px; text-align:center;'>
    <h1 style='color:#5e928a; font-size:32px; margin-bottom:5px;'>Exploratory Data Analysis</h1>
    <p style='color:{HEADER_COLOR}; font-size:20px; margin-top:0px;'>
        This section explores the behavioral patterns in the dataset — uncovering how lifestyle, biological, and environmental factors relate to hair loss. The goal is to identify visible trends and possible predictors through visual analysis.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #DDD;'>", unsafe_allow_html=True)

# ======== LOAD DATASETS ==========
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

df_predict_cleaned = load_csv("Data/Predict Hair Fall Cleaned.csv")
df_luke_cleaned = load_csv("Data/Luke_hair_loss_documentation Cleaned.csv")

# ======== DATASET SELECTOR ==========
dataset_option = st.selectbox(
    "Select Dataset:",
    ["None", "Hair Health Prediction Dataset", "Luke Hair Loss Dataset"],
    index=0
)

if dataset_option == "None":
    st.info("Select a dataset from the dropdown above to begin.")
else:
    # ===== Dataset Banner =====
    st.markdown(
        f"<div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:12px;'>"
        f"<h2 style='color:{ACCENT}; font-size:26px; margin:5px 0 5px 0;'>{dataset_option}</h2></div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


# ---- Hair Health Prediction Dataset Bar Plot ----
if dataset_option == "Hair Health Prediction Dataset":
    df = df_predict_cleaned.copy()

    st.markdown(f"""
    <div padding:14px; border-radius:12px;'>
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
    This section explores the overall distribution of key features in the dataset. Understanding how factors like <b>Stress Level</b>, <b>Age Range</b>, and <b>Environmental Conditions</b> are spread across the population helps reveal underlying data patterns and potential biases. 
    Interactive visualizations below allow you to observe how each attribute varies among individuals.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Feature Distributions</h3> </div> """, unsafe_allow_html=True) 
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
   <p style='font-size:18px; color:{TEXT}; margin:0;'>
    This section visualizes the distribution of individual features to build intuition about the dataset.
    Use the selector below to choose any variable — each plot shows simple counts for that feature so you can quickly assess skew, balance, and rare categories.
    Start with the default view (Hair_Loss) to see how severity is distributed across the population.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    palette = ["#bddb8e", "#85ada6", "#A8D5BA", "#E67E22", "#FFB347", "#D9D9D9"]

    if 'df_predict_cleaned' not in globals() or df_predict_cleaned is None:
        st.error("Cleaned Hair Health dataset not loaded.")
    else:
        df = df_predict_cleaned.copy()


        # default choice
        palette = ["#bddb8e", "#85ada6", "#A8D5BA", "#E67E22", "#FFB347", "#D9D9D9"]

        if 'df_predict_cleaned' not in globals() or df_predict_cleaned is None:
            st.error("Cleaned Hair Health dataset not loaded.")
        else:
            df = df_predict_cleaned.copy()

            default_col = 'Hair_Loss' if 'Hair_Loss' in df.columns else df.columns[0]
            st.markdown(f"<p style='font-size:18px; color:{TEXT};'><b>Select feature to plot (counts):</b></p>", unsafe_allow_html=True)
            col_choice = st.selectbox("Feature:", options=list(df.columns), index=list(df.columns).index(default_col))

            try:
                # prepare counts (treat NaN as 'Missing')
                counts = df[col_choice].fillna('Missing').astype(str).value_counts().sort_index()
                categories = counts.index.tolist()
                values = counts.values

                # color mapping (cycle palette if needed)
                colors = [palette[i % len(palette)] for i in range(len(categories))]

                # create Plotly bar chart
                fig = go.Figure(
                    data=go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
                    )
                )

                fig.update_layout(
                    title=dict(text=f"Count Distribution — {col_choice}", x=0.35, font=dict(size=23)),
                    xaxis_title=col_choice,
                    yaxis_title="Count",
                    template="simple_white",
                    margin=dict(l=40, r=20, t=60, b=120),
                    xaxis_tickangle=-45,
                    hovermode="closest",
                    height=850,  # ⬆️ Taller chart
                    width=250,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(
                        showgrid=True,  # vertical gridlines
                        gridcolor='rgba(0,0,0,0.1)',
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,  # horizontal gridlines
                        gridcolor='rgba(0,0,0,0.15)',
                        zeroline=False
                    )
                )

                # allow zoom/pan and show modebar
                config = {
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": ["drawrect", "drawopenpath", "eraseshape"],
                    "displaylogo": False,
                }

                # If many categories, enable horizontal scrolling by increasing width via container
                st.plotly_chart(fig, use_container_width=True, config=config)

            except Exception as e:
                st.error("Graph cannot be created.")
                print(f"Error creating interactive distribution plot for {col_choice}: {e}")

        
        st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Relationships Between Variables</h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
            When analyzing relationships, certain trends become apparent — higher stress levels and genetics are associated with greater reported hair loss. These associations suggest that both emotional and biological stressors may play a role.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)     

# ---- Two-variable interactive explorer ----
        palette = ["#bddb8e", "#85ada6", "#A8D5BA", "#E67E22", "#FFB347", "#D9D9D9"]

        if 'df_predict_cleaned' not in globals() or df_predict_cleaned is None:
            st.error("Cleaned Hair Health dataset not loaded.")
        else:
            df = df_predict_cleaned.copy()

            # default choices
            default_x = 'Stress' if 'Stress' in df.columns else df.columns[0]
            default_y = 'Hair_Loss' if 'Hair_Loss' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]

            st.markdown(f"<p style='font-size:18px; color:{TEXT};'><b>Select two variables to explore their relationship:</b></p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                x_var = st.selectbox("X variable", options=list(df.columns), index=list(df.columns).index(default_x))
            with col2:
                y_var = st.selectbox("Y variable", options=list(df.columns), index=list(df.columns).index(default_y))

            st.markdown("<br>", unsafe_allow_html=True)

            try:
            # Fill NaNs for categorical variables
                temp = df[[x_var, y_var]].copy()
                temp = temp.fillna('Missing')

                # If either variable is Hair_Loss, treat it as categorical for counts
                if x_var == "Hair_Loss" or y_var == "Hair_Loss":
                    temp[x_var] = temp[x_var].astype(str)
                    temp[y_var] = temp[y_var].astype(str)
                    counts = temp.groupby([x_var, y_var]).size().reset_index(name='count')

                    fig = px.bar(
                        counts,
                        x=x_var,
                        y='count',
                        color=y_var,
                        barmode='group',
                        color_discrete_sequence=palette,
                        labels={'count': 'Count'},
                        title=f"Counts of {y_var} by {x_var}" if y_var == "Hair_Loss" else f"Counts of {x_var} by {y_var}"
                    )
                    fig.update_layout(xaxis_tickangle=-45, margin=dict(t=60, b=140))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Default for numeric vs numeric or numeric vs categorical
                    is_x_num = pd.api.types.is_numeric_dtype(temp[x_var])
                    is_y_num = pd.api.types.is_numeric_dtype(temp[y_var])

                    if not is_x_num and not is_y_num:
                        # Both categorical → counts
                        counts = temp.groupby([x_var, y_var]).size().reset_index(name='count')
                        fig = px.bar(
                            counts,
                            x=x_var,
                            y='count',
                            color=y_var,
                            barmode='group',
                            color_discrete_sequence=palette,
                            labels={'count': 'Count'},
                            title=f"{y_var} vs {x_var}"
                        )
                        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=60, b=140))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # One or both numeric → scatter or box
                        fig = px.scatter(temp, x=x_var, y=y_var, trendline=None, labels={x_var:x_var, y_var:y_var})
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("Graph cannot be created.")
                print(f"Error creating plot for {x_var} vs {y_var}: {e}")

        st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Correlations and Insights</h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
            Understanding correlations between features helps uncover patterns that might explain hair loss. 
            For example, strong positive correlations between stress-related features and hair loss indicate potential emotional or lifestyle influences, 
            while weak correlations with other factors suggest limited impact. 
            This heatmap highlights these relationships and guides further analysis.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        df_predict_raw = load_csv("Data/Predict Hair Fall Raw.csv")
        
        df_corr = df_predict_raw.copy()
        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = pd.factorize(df_corr[col])[0]

        # Compute correlation matrix
        corr_matrix = df_corr.corr()

        # Plot heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='YlGnBu',
            labels=dict(x="Features", y="Features", color="Correlation"),
            title="Correlation Matrix Heatmap"
        )
        fig.update_layout(
            width=1600,   # set width in pixels
            height=1200,   # set height in pixels
            margin=dict(t=60, b=60, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)


                
