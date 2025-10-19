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
SECTION_BG_PLOTS = "#749683"
SIDBAR_TEXT = "#a9cac6"
CARD_COLOR ="#d4e6e4"
CARD_COLOR2 = "#dcf4e0"

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
    
    st.markdown(f""" <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Feature Distributions</h3> </div> """, unsafe_allow_html=True) 
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
                colors = ["#A8D5BA", "#4c8179", "#A8D5BA", "#5d9189", "#2E8B57"]

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

        
        st.markdown(f""" <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Relationships Between Variables</h3> </div> """, unsafe_allow_html=True) 
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

        st.markdown(f""" <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Correlations and Insights</h3> </div> """, unsafe_allow_html=True) 
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
        
        # ===== Genetics Insight Section =====
        st.markdown(f"""
        <div style='background-color:#2a5a55; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Useful Findings</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Prepare data for stacked bar
        changeWithTarget = df.groupby(['Genetic_Encoding', 'Hair_Loss']).size().unstack(fill_value=0)

        # Calculate proportions
        prop_with_genetic = changeWithTarget.loc[1] / changeWithTarget.loc[1].sum() * 100
        prop_without_genetic = changeWithTarget.loc[0] / changeWithTarget.loc[0].sum() * 100
        
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR};; padding:12px 20px; border-left:6px solid #2E8B57; border-radius:10px; margin-top:20px;'>
            <p style='font-size:20px; color:{TEXT}; margin:0;'>
                <b>1) Genetics and Hair Loss Insight</b><br><br>
                Genetics plays a major role in hair loss. People without a genetic predisposition report much lower hair loss, 
                whereas among people with a genetic predisposition, a significantly higher proportion report hair loss. 
                This confirms that family history is a strong risk factor.
            </p>
            <br>
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
                <b>Hair loss proportion among people WITH genetic predisposition:</b><br>{prop_with_genetic.to_frame(name='proportion')}
            </p>
            <br>
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
                <b>Hair loss proportion among people WITHOUT genetic predisposition:</b><br>{prop_without_genetic.to_frame(name='proportion')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10,6))
        changeWithTarget.plot(
            kind='bar',
            stacked=True,
            alpha=0.9,
            ax=ax,
            color = ["#bddb8e", "#85ada6", "#A8D5BA", "#5d9189", "#2E8B57"]   # color scheme
        )
        ax.set_title("Counts of Genetic_Encoding by Hair Loss", fontsize=16, pad=15)
        ax.set_xlabel('Genetic_Encoding', fontsize=14, labelpad=10)
        ax.set_ylabel("Count", fontsize=14, labelpad=10)
        ax.legend(title="Hair Loss", labels=["No Hair Loss", "Hair Loss"], fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
        
        colors = ["#85ada6", "#5d9189", "#A8D5BA", "#5d9189", "#2E8B57"] 
        # Filter dataset for genetically predisposed individuals
        df_genetic = df_predict_cleaned[df_predict_cleaned['Genetic_Encoding'] == 1]

        # --- Step 2: Group by Nutritional Deficiency and Hair_Loss ---
        counts = df_genetic.groupby(['Nutritional_Deficiencies', 'Hair_Loss']).size().reset_index(name='Count')

        # --- Step 3: Convert Hair_Loss to string for hue ---
        counts['Hair_Loss'] = counts['Hair_Loss'].astype(str)

        # --- Step 4: Display card ---
        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:12px 20px; border-left:6px solid #2E8B57; border-radius:10px; margin-top:20px;'>
            <p style='font-size:20px; color:{TEXT}; margin:0;'>
                <b>2) Nutritional Deficiencies and Hair Loss</b><br><br>
                Nutritional deficiencies further modulate hair loss among genetically predisposed individuals. 
                For example, iron deficiency is present in 54% of people reporting hair loss with genetic predisposition,
                and Vitamin D deficiency in 58%, highlighting the importance of these nutrients for hair strength and growth.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        green_palette = ["#bddb8e", "#85ada6"] 
        
        # --- Step 5: Plot ---
        try:
            fig, ax = plt.subplots(figsize=(14,6))
            sns.barplot(
                data=counts,
                x='Nutritional_Deficiencies',
                y='Count',
                hue='Hair_Loss',
                ax=ax,
                palette=green_palette  # use palette instead of color
            )
            ax.set_title('Hair Loss Distribution by Nutritional Deficiency (Genetic Hair Loss = 1)', fontsize=16, pad=15)
            ax.set_xlabel('Nutritional Deficiencies', fontsize=14, labelpad=10)
            ax.set_ylabel('Number of People', fontsize=14, labelpad=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
            ax.legend(title='Hair Loss', labels=['No (0)', 'Yes (1)'], fontsize=12)
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Graph cannot be created.")
        
        st.markdown("<br>", unsafe_allow_html=True)
            
        # --- Step 1: Create age bins ---
        bins = [18, 25, 40, 51]
        labels = ['18-25', '25-40', '40-51']
        df_predict_cleaned['Age_Range_New'] = pd.cut(df_predict_cleaned['Age'], bins=bins, labels=labels, right=False)

        # --- Step 2: Filter for Alopecia conditions ---
        alopecia_conditions = ['Alopecia Areata', 'Androgenetic Alopecia']
        df_alopecia = df_predict_cleaned[df_predict_cleaned['Medical_Conditions'].isin(alopecia_conditions)]

        # --- Step 3: Group by new age range and condition, summing hair loss ---
        alopecia_counts = df_alopecia.groupby(['Age_Range_New', 'Medical_Conditions'])['Hair_Loss'].sum().unstack(fill_value=0)

        # --- Step 4: Display observation card ---
        # Convert alopecia_counts to HTML table
        alopecia_table_html = alopecia_counts.to_html(classes='table', border=0, index=True)

        st.markdown(f"""
        <div style='background-color:{CARD_COLOR}; padding:12px 20px; border-left:6px solid #2E8B57; border-radius:10px; margin-top:20px;'>
            <p style='font-size:20px; color:{TEXT}; margin:0;'>
                <b>3) Alopecia and Age Trends</b><br><br>
                As expected, individuals with Alopecia (Alopecia Areata and Androgenetic Alopecia) report higher hair loss. 
                Androgenetic Alopecia, strongly linked to genetics (male pattern baldness), is most prevalent in people with a family history of hair fall. 
                This pattern is also reflected in age trends: individuals aged 25-40 report the highest number of hair loss cases, 
                consistent with the peak occurrence of Androgenetic Alopecia.
            </p>
            <br>
            <p style='font-size:20px; color:{TEXT}; margin:0;'>
                <b>Hair Loss Counts by Age Range for Alopecia Types:</b>
            </p>
            {alopecia_table_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Filter genetic + alopecia conditions
        alopecia_conditions = ['Alopecia Areata', 'Androgenetic Alopecia']
        df_genetic = df_predict_cleaned[(df_predict_cleaned['Genetic_Encoding'] == 1) & 
                            (df_predict_cleaned['Medical_Conditions'].isin(alopecia_conditions))]

        # Group by Alopecia type and Hair_Loss
        counts = df_genetic.groupby(['Medical_Conditions', 'Hair_Loss']).size().reset_index(name='Count')
        counts['Hair_Loss'] = counts['Hair_Loss'].astype(str)  # Convert Hair_Loss to string for hue

        st.markdown("<br>", unsafe_allow_html=True)
        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=counts, x='Medical_Conditions', y='Count', hue='Hair_Loss', palette=["#a4d4be", "#85ada6"], ax=ax)
        ax.set_title('Hair Loss by Alopecia Type (Genetic Encoding = 1)', fontsize=9, pad=15)
        ax.set_xlabel('Alopecia Type', fontsize=9)
        ax.set_ylabel('Number of People', fontsize=9)
        ax.legend(title='Hair Loss', labels=['No (0)', 'Yes (1)'], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)


        
        # --- Step 5: Plot ---
        try:
            fig, ax = plt.subplots(figsize=(10,6))
            alopecia_counts.plot(kind='bar', ax=ax, color=["#a4d4be", "#85ada6"])
            ax.set_title('Hair Loss by Age Range for Alopecia Conditions', fontsize=9)
            ax.set_ylabel('Number of People Reporting Hair Loss', fontsize=9)
            ax.set_xlabel('Age Range', fontsize=9)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend(title='Alopecia Type')
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Graph cannot be created.")
elif dataset_option == "Luke Hair Loss Dataset":
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
    
    if 'df_luke_cleaned' not in globals() or df_luke_cleaned is None:
        st.error("Luke Hair Dataset not loaded.")
    else:
        df = df_luke_cleaned.copy()         
        





                                        
