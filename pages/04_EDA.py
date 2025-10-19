import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
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
        requested_cols = ['Hair_Loss', 'Genetic_Encoding', 'Hormonal_Changes', 'Stress_Level', 'Age', 'Smoking', 'Weight_Loss']

# load raw df (you already have df_predict_raw from load_csv)
        df_corr_src = df_predict_cleaned.copy()

        # pick only columns that actually exist in the dataframe
        cols = [c for c in requested_cols if c in df_corr_src.columns]
        if not cols:
            st.error("None of the requested columns were found in the dataset: " + ", ".join(requested_cols))
        else:
            # Make a working copy and encode non-numeric columns
            df_work = df_corr_src[cols].copy()
            for col in df_work.columns:
                if df_work[col].dtype == 'object' or df_work[col].dtype.name == 'category':
                    # factorize (preserves ordering of appearance)
                    df_work[col] = pd.factorize(df_work[col].astype(str))[0]

            # compute correlation matrix
            corr_matrix = df_work.corr()

            # green palette (your chosen greens)
            colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
            cmap = LinearSegmentedColormap.from_list("green_palette", colors_palette, N=256)

            # plot
            plt.figure(figsize=(10, 8))
            sns.set(font_scale=1.0)
            ax = sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=-1,
                vmax=1,
                square=False,
                linewidths=0.8,
                linecolor='white',
                cbar_kws={'shrink': 0.7, 'pad': 0.02}
            )

            ax.set_title("Correlation Matrix — Selected Features (Predict Dataset)", fontsize=16, color=HEADER_COLOR, pad=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            st.pyplot(plt.gcf())
            plt.close()

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
        default_col = 'Hair_Loss'     
        col_choice = st.selectbox("Feature:", 
                          options=['Date', 'Hair_Loss', 'Stay_Up_Late', 'Pressure_Level', 
                                   'Coffee_Consumed', 'Brain_Working_Duration', 'School_Assesssment', 
                                   'Stress_Level', 'Shampoo_Brand', 'Swimming', 'Hair_Washing', 
                                   'Hair_Grease', 'Dandruff', 'Libido'], 
                          index=1 if default_col=='Hair_Loss' else 0)

# Define custom order for categorical features
        custom_orders = {
            'Hair_Loss': ['Few', 'Medium', 'Many', 'A lot'],
            'Pressure_Level': ['Low', 'Medium', 'High', 'Very High'],
            'Stress_Level': ['Low', 'Medium', 'High', 'Very High'],
            'Dandruff': ['None', 'Few', 'Many'],
            'Hair_Washing': ['No', 'Yes'],
            'Swimming': ['No', 'Yes'],
            'School_Assesssment': ['No assessment', 'Individual Assessment', 'Team Assessment', 'Final exam', 'Final exam revision' ]
            
        }

        try:
            # prepare counts
            col_data = df[col_choice].fillna('Missing').astype(str)
            if col_choice in custom_orders:
                categories = custom_orders[col_choice]
                counts = col_data.astype(str).value_counts().reindex(categories, fill_value=0)
            elif col_choice in ['Coffee_Consumed', 'Brain_Working_Duration', 'Stay_Up_Late']:
                col_numeric = pd.to_numeric(col_data, errors='coerce')
                counts = col_numeric.value_counts().sort_index()
                categories = counts.index.tolist()
            else:
                # for other categorical columns
                counts = col_data.astype(str).value_counts().sort_index()
                categories = counts.index.tolist()
            
            values = counts.values

            # Colors palette
            colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
            colors = [colors_palette[i % len(colors_palette)] for i in range(len(categories))]

            # Plotly bar chart
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
                height=850,
                width=250,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.15)', zeroline=False)
            )

            config = {
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawrect", "drawopenpath", "eraseshape"],
                "displaylogo": False,
            }

            st.plotly_chart(fig, use_container_width=True, config=config)

        except Exception as e:
            st.error("Graph cannot be created.")
            print(f"Error creating interactive distribution plot for {col_choice}: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Relationship Between Variables</h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
            This section explores how different features interact with each other and with <b>Hair_Loss</b>. 
            Interactive visualizations allow you to examine trends and patterns — for instance, whether higher <b>Coffee Consumption</b> or certain <b>Brain Working Duration</b> are associated with increased hair loss. 
            Use the selectors below to choose any pair of variables and investigate their relationships.
        </p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) 
        
        
        custom_orders = {
        'Hair_Loss': ['Few', 'Medium', 'Many', 'A lot'],
        'Pressure_Level': ['Low', 'Medium', 'High', 'Very High'],
        'Stress_Level': ['Low', 'Medium', 'High', 'Very High'],
        'Dandruff': ['None', 'Few', 'Many'],
        'Hair_Washing': ['No', 'Yes'],
        'Swimming': ['No', 'Yes'],
        'School_Assesssment': ['No assessment', 'Individual Assessment', 'Team Assessment', 'Final exam', 'Final exam revision' ]
    }

# ===== Select Variables =====
        default_x = 'Stay_Up_Late'
        default_y = 'Hair_Loss'

        x_choice = st.selectbox("X-axis:", options=df.columns.tolist(), index=df.columns.get_loc(default_x))
        y_choice = st.selectbox("Y-axis:", options=df.columns.tolist(), index=df.columns.get_loc(default_y))

        try:
            # Prepare data
            df_plot = df[[x_choice, y_choice]].copy()
            df_plot[y_choice] = df_plot[y_choice].fillna('Missing')
            df_plot[x_choice] = df_plot[x_choice].fillna('Missing')

            # ===== Determine categories for X =====
            col_data = df_plot[x_choice]
            if x_choice in custom_orders:
                categories_x = custom_orders[x_choice]
                counts_x = col_data.astype(str).value_counts().reindex(categories_x, fill_value=0)
            elif x_choice in ['Coffee_Consumed', 'Brain_Working_Duration', 'Stay_Up_Late']:
                col_numeric = pd.to_numeric(col_data, errors='coerce')
                counts_x = col_numeric.value_counts().sort_index()
                categories_x = counts_x.index.tolist()
            else:
                counts_x = col_data.astype(str).value_counts().sort_index()
                categories_x = counts_x.index.tolist()

            # ===== Determine categories for Y =====
            col_data_y = df_plot[y_choice].astype(str)
            if y_choice in custom_orders:
                categories_y = custom_orders[y_choice]
            else:
                categories_y = col_data_y.unique().tolist()

            # ===== Prepare counts for stacked bars =====
            counts = df_plot.groupby([x_choice, y_choice]).size().reset_index(name='Count')

            # ===== Green-based distinct palette =====
            colors_palette = ["#b7e4c7", "#2db17a", "#186b46", "#d9a044", "#74c69d", "#c744c1"]

            # ===== Create stacked bar chart =====
            fig = go.Figure()
            for i, cat in enumerate(categories_y):
                cat_data = counts[counts[y_choice] == cat]
                # Align x values with categories
                cat_counts = [cat_data.loc[cat_data[x_choice] == x_val, 'Count'].sum() if x_val in cat_data[x_choice].values else 0 for x_val in categories_x]
                fig.add_trace(go.Bar(
                    x=categories_x,
                    y=cat_counts,
                    name=str(cat),
                    marker_color=colors_palette[i % len(colors_palette)],
                    hovertemplate=f"{y_choice}: {cat}<br>{x_choice}: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))

            fig.update_layout(
                barmode='stack',
                title=f"{y_choice} vs {x_choice}",
                xaxis_title=x_choice,
                yaxis_title="Count",
                template="simple_white",
                height=600,
                hovermode="closest"
            )

            config = {
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawrect", "drawopenpath", "eraseshape"],
                "displaylogo": False,
            }
            st.plotly_chart(fig, use_container_width=True, config=config)

        except Exception as e:
            st.error("Graph cannot be created.")
            print(f"Error creating interactive 2-variable plot: {e}")

        st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Correlations and Insights</h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
            This section explores correlations between numeric variables and key features in the dataset. 
            Positive or negative relationships can highlight potential drivers of hair loss, stress levels, and other important outcomes. 
            Use the heatmap below to quickly identify strong associations.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        
        cols = ['Date', 'Hair_Loss_Encoding', 'Stay_Up_Late', 'Pressure_Level_Encoding', 
        'Coffee_Consumed', 'Brain_Working_Duration', 'School_Assesssment', 
        'Stress_Level_Encoding', 'Shampoo_Brand', 'Swimming', 'Hair_Washing', 
        'Hair_Grease', 'Dandruff_Encoding', 'Libido']

        df_corr = df_luke_cleaned[cols].copy()

        # Encode categorical columns
        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))

        # Compute correlation
        corr_matrix = df_corr.corr()

        # Define green palette
        colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
        cmap = LinearSegmentedColormap.from_list("green_palette", colors_palette, N=256)

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True, linewidths=0.8, linecolor='white')
        plt.title("Correlation Heatmap — Luke Dataset", fontsize=16, color="#2E8B57")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close() 
        
        # --- HAIR LOSS OVER TIME --- 
        st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Hair Loss Over Time (Smoothened)</h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
            Using this time series plot, we can observe that there is no consistent trend in hair fall over time. 
            Any increases or decreases appear to depend on other factors such as <b>Coffee Consumption</b>, <b>Staying Up Late</b>, or <b>Stress Levels</b>.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        df_luke_cleaned['Date'] = pd.to_datetime(df_luke_cleaned['Date'], format='%d/%m/%Y')

# Sort by date
        df = df_luke_cleaned.sort_values('Date').copy()

        # Apply rolling average to smooth the Hair_Loss_Encoding
        df['Smoothed_Hair_Loss'] = df['Hair_Loss_Encoding'].rolling(window=5, center=True, min_periods=1).mean()

        # Plot interactive time series
        fig = px.line(
            df, 
            x='Date', 
            y='Smoothed_Hair_Loss', 
            labels={'Smoothed_Hair_Loss': 'Hair Loss (Smoothed)', 'Date':'Date'},
            line_shape='spline',
            template='simple_white'
        )

        fig.update_traces(line=dict(color='mediumseagreen', width=3), hovertemplate='Date: %{x|%d-%b-%Y}<br>Hair Loss: %{y:.2f}<extra></extra>')
        fig.update_layout(height=500, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown(f""" <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Useful Insights </h3> </div> """, unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        The analysis reveals that <b>Hair Loss</b> has a positive correlation with several lifestyle and behavioral factors such as 
        <b>Staying Up Late</b>, <b>Long Brain Working Duration</b>, <b>Hair Grease</b>, <b>Coffee Consumed</b>, <b>Pressure Level</b>, 
        and <b>Stress Level</b>. 
        <br><br>
        This means that whenever Luke experienced higher stress or pressure levels, work for extended periods, consume more coffee, 
        or frequently stay up late, the likelihood of hair loss tends to increase. 
        Similarly, higher hair grease buildup — which may block follicles — also appears linked to increased hair shedding. 
        Overall, these findings suggest that both <b>lifestyle habits</b> and <b>mental well-being</b> play a significant role in influencing hair health.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        cols = ['Hair_Loss_Encoding', 'Stay_Up_Late', 'Pressure_Level_Encoding', 'Coffee_Consumed', 
        'Brain_Working_Duration', 'Stress_Level_Encoding', 
        'Shampoo_Brand', 'Swimming', 'Hair_Washing', 'Hair_Grease', 
        'Dandruff_Encoding', 'Libido']

        df_corr = df_luke_cleaned[cols].copy()

        # Encode categorical columns for correlation
        from sklearn.preprocessing import LabelEncoder

        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))

        # Compute correlations
        corr_matrix = df_corr.corr()

        # Extract correlations with Hair_Loss only
        hair_loss_corr = corr_matrix['Hair_Loss_Encoding'].drop('Hair_Loss_Encoding').sort_values(ascending=False)

        # Create interactive horizontal bar chart
        colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
        fig = px.bar(
            x=hair_loss_corr.values,
            y=hair_loss_corr.index,
            orientation='h',
            color=hair_loss_corr.values,
            color_continuous_scale=colors_palette,
            labels={'x':'Correlation with Hair_Loss', 'y':'Feature'},
            text=hair_loss_corr.values.round(2)
        )

        fig.update_layout(
            title=dict(text="Feature Correlation with Hair_Loss", x=0.35, font=dict(size=22)),
            xaxis=dict(range=[-1,1]),
            height=600,
            template="simple_white"
        )

        fig.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>Correlation: %{x:.2f}<extra></extra>')

        st.plotly_chart(fig, use_container_width=True)
        
        
        # --- Section Title ---
        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:40px;'> 
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Bubble Chart of Hair Loss vs Second Variable with Third Variable as Size & Color</h3> 
        </div> 
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Description ---
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        This interactive <b>Bubble Chart</b> helps explore how <b>two lifestyle or behavioral factors</b> jointly influence hair loss. 
        Each point represents an observation, where:
        <br>• The <b>x-axis</b> shows the chosen second variable,  
        <br>• The <b>y-axis</b> represents the <b>Hair Loss Encoding</b> (severity of hair loss),  
        <br>• The <b>bubble size and color</b> correspond to a third variable of your choice.  
        <br><br>
        By analyzing this visualization, you can identify how combinations — for instance, <b>higher coffee consumption and late nights</b> — 
        correlate with greater hair loss intensity.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Dropdowns for dynamic variable selection ---
        second_var = st.selectbox(
            "Select the second variable (X-axis):",
            ['Stay_Up_Late', 'Coffee_Consumed', 'Brain_Working_Duration', 'Stress_Level', 
            'Pressure_Level', 'Hair_Grease', 'Dandruff', 'Libido'],
            index=1
        )

        third_var = st.selectbox(
            "Select the third variable (Bubble Size & Color):",
            ['Coffee_Consumed', 'Stay_Up_Late', 'Brain_Working_Duration', 'Stress_Level_Encoding', 
            'Pressure_Level_Encoding', 'Hair_Grease', 'Dandruff_Encoding', 'Libido'],
            index=2
        )

        # --- Bubble Chart ---
        try:
            fig3 = px.scatter(
                df,
                x=second_var,
                y='Hair_Loss_Encoding',
                size=third_var,
                color=third_var,
                color_continuous_scale='Viridis',
                hover_data=['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'],
                title=f'Bubble Chart of Hair Loss vs {second_var} with {third_var} as Size & Color'
            )

            fig3.update_layout(
            template="simple_white",
            title_x=0.05,
            title_font=dict(size=20),
            xaxis_title=second_var,
            yaxis_title="Hair Loss Encoding",
            hovermode="closest",
            height=700,
            margin=dict(l=40, r=20, t=80, b=80),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False,
                linecolor='rgba(0,0,0,0.3)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False,
                linecolor='rgba(0,0,0,0.3)'
            )
        )


            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error("Could not generate the bubble chart.")
            print(f"Error generating bubble chart: {e}")


        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:40px;'> 
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Bubble Chart of Hair Loss with Multi-Variable Interaction</h3> 
        </div> 
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Description ---
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        This interactive <b>4-variable bubble chart</b> explores how multiple lifestyle or behavioral factors jointly influence hair loss.  
        Each bubble represents an individual data point:
        <br>• <b>X-axis:</b> Second variable of interest  
        <br>• <b>Y-axis:</b> Hair Loss Encoding (severity)  
        <br>• <b>Bubble size:</b> Third variable  
        <br>• <b>Bubble color:</b> Fourth variable  
        <br><br>
        For example, you can see how <b>late-night habits (Stay Up Late)</b> and <b>coffee consumption</b> interact,  
        with <b>stress levels</b> represented through color intensity.  
        This view helps uncover <b>combined effects</b> that individual correlations might miss.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Dropdowns for dynamic selection ---
        second_var = st.selectbox(
            "Select the second variable (X-axis):",
            ['Stay_Up_Late', 'Coffee_Consumed', 'Brain_Working_Duration', 'Stress_Level', 
            'Pressure_Level', 'Hair_Grease', 'Dandruff', 'Libido'],
            index=4
        )

        third_var = st.selectbox(
            "Select the third variable (Bubble Size):",
            ['Coffee_Consumed', 'Stay_Up_Late', 'Brain_Working_Duration', 'Stress_Level_Encoding', 
            'Pressure_Level_Encoding', 'Hair_Grease', 'Dandruff_Encoding', 'Libido'],
            index=5
        )

        fourth_var = st.selectbox(
            "Select the fourth variable (Bubble Color):",
            ['Coffee_Consumed', 'Stay_Up_Late', 'Brain_Working_Duration', 'Stress_Level_Encoding', 
            'Pressure_Level_Encoding', 'Hair_Grease', 'Dandruff_Encoding', 'Libido'],
            index=6
        )

        # --- Bubble Chart ---
        try:
            fig4 = px.scatter(
                df,
                x=second_var,
                y='Hair_Loss_Encoding',
                size=third_var,
                color=fourth_var,
                color_continuous_scale='Viridis',
                hover_data=['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'],
                title=f'Bubble Chart of Hair Loss vs {second_var} (Size = {third_var}, Color = {fourth_var})'
            )

            # --- Gridline & Style Customization ---
            fig4.update_layout(
                template="simple_white",
                title_x=0,
                title_font=dict(size=22),
                xaxis_title=second_var,
                yaxis_title="Hair Loss Encoding",
                hovermode="closest",
                height=750,
                margin=dict(l=50, r=20, t=80, b=80),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=False,
                    linecolor='rgba(0,0,0,0.3)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=False,
                    linecolor='rgba(0,0,0,0.3)'
                )
            )

            st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error("Could not generate the 4-variable bubble chart.")
            print(f"Error generating bubble chart: {e}")
        

        # Optional controls
        opacity = 1       # set bubble opacity
        sample_n = None      # set to an int like 1000 to sample points for performance, or None to use all

        # ---------- Section A (single third variable = z, size, color) ----------
        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:30px;'> 
            <h3 style='color:{ACCENT}; font-size:22px; margin:6px 0;'>3D Scatter: Hair Loss vs Second Variable (Size & Color = Third Variable)</h3> 
        </div> 
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        

        st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
            This interactive <b>3D scatter plot</b> provides a deeper understanding of how multiple variables together influence hair loss patterns.  
            Each point represents a data observation, and its visual properties carry distinct meanings:
            <br>• <b>X-axis:</b> Second variable (e.g., Stay Up Late or Coffee Consumed)  
            <br>• <b>Y-axis:</b> Hair Loss Encoding (severity level)  
            <br>• <b>Z-axis, Bubble Size, and Color:</b> Third variable of interest  
            <br><br>
            By combining <b>magnitude</b> (bubble size) and <b>intensity</b> (color), you can visually interpret how certain factors interact.  
            For instance, a strong clustering of large, darker bubbles might indicate that <b>increased coffee consumption and higher stress levels</b>  
            coincide with greater hair loss severity.  
            This visualization helps uncover <b>non-linear, multi-variable relationships</b> that are not visible in simple pairwise comparisons.
            </p>
            """, unsafe_allow_html=True)

        
        st.markdown("<br>", unsafe_allow_html=True)

        second_var_a = st.selectbox("Section A — Select X variable", 
                                    options=['Stay_Up_Late','Coffee_Consumed','Brain_Working_Duration','Stress_Level',
                                            'Pressure_Level','Hair_Grease','Dandruff','Libido'],
                                    index=0, key="secA_x")
        third_var_a = st.selectbox("Section A — Select Z / Size / Color variable", 
                                options=['Stay_Up_Late', 'Coffee_Consumed', 'Brain_Working_Duration', 'Stress_Level_Encoding', 
                                         'Pressure_Level_Encoding', 'Hair_Grease', 'Dandruff_Encoding', 'Libido'],
                                index=1, key="secA_z")

        try:
            # build tmp with required columns (ensure Hair_Loss included for hover)
            needed_cols = [second_var_a, 'Hair_Loss_Encoding', third_var_a, 'Hair_Loss']
            # include only columns that actually exist in df
            needed_cols = [c for c in needed_cols if c in df.columns]
            tmp = df[needed_cols].copy()

            # coerce numeric where needed
            for col in [second_var_a, 'Hair_Loss_Encoding', third_var_a]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')

            # sampling for performance (optional)
            if sample_n is not None and len(tmp) > sample_n:
                tmp = tmp.sample(sample_n, random_state=42)

            # drop rows missing the core axes
            core_axes = [c for c in [second_var_a, 'Hair_Loss_Encoding', third_var_a] if c in tmp.columns]
            tmp = tmp.dropna(subset=core_axes)

            # determine hover columns that exist
            hover_cols = [c for c in ['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'] if c in tmp.columns]

            fig_a = px.scatter_3d(
                tmp,
                x=second_var_a if second_var_a in tmp.columns else None,
                y='Hair_Loss_Encoding' if 'Hair_Loss_Encoding' in tmp.columns else None,
                z=third_var_a if third_var_a in tmp.columns else None,
                size=third_var_a if third_var_a in tmp.columns else None,
                color=third_var_a if third_var_a in tmp.columns else None,
                color_continuous_scale='Viridis',
                hover_data=hover_cols,
                title=f'\n\nHair Loss vs {second_var_a} vs {third_var_a} (size & color = {third_var_a})',
                opacity=opacity
            )

            fig_a.update_layout(
                height=700,
                scene=dict(
                    xaxis_title=second_var_a,
                    yaxis_title='Hair_Loss_Encoding',
                    zaxis_title=third_var_a,
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
                    zaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
                ),
                margin=dict(l=20, r=20, t=70, b=20)
            )

            st.plotly_chart(fig_a, use_container_width=True)

        except Exception as e:
            st.error("Could not generate Section A 3D chart. See console for details.")
            print("Section A error:", e)


        #