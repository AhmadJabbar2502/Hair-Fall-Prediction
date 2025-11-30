# /EDA/Predict_Hair_Fall.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

# Local copies of the style colors used in the main page
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

def render_predict_page(df_predict_cleaned):

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

    if df_predict_cleaned is None:
        st.error("Cleaned Hair Health dataset not loaded.")
        return
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
                height=850,
                width=250,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.15)',
                    zeroline=False
                )
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

    # ---------------- Relationships Between Variables ----------------
    st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Relationships Between Variables</h3> </div> """, unsafe_allow_html=True) 
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
            When analyzing relationships, certain trends become apparent — higher stress levels and genetics are associated with greater reported hair loss. These associations suggest that both emotional and biological stressors may play a role.
            </p>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)     

    # ---- Two-variable interactive explorer ----
    if df_predict_cleaned is None:
        st.error("Cleaned Hair Health dataset not loaded.")
        return
    else:
        df = df_predict_cleaned.copy()

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
            temp = df[[x_var, y_var]].copy()
            temp = temp.fillna('Missing')

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
                is_x_num = pd.api.types.is_numeric_dtype(temp[x_var])
                is_y_num = pd.api.types.is_numeric_dtype(temp[y_var])

                if not is_x_num and not is_y_num:
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
                    fig = px.scatter(temp, x=x_var, y=y_var, trendline=None, labels={x_var:x_var, y_var:y_var})
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Graph cannot be created.")
            print(f"Error creating plot for {x_var} vs {y_var}: {e}")

    # ---------------- Correlations and Insights ----------------
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

    df_corr_src = df_predict_cleaned.copy()

    cols = [c for c in requested_cols if c in df_corr_src.columns]
    if not cols:
        st.error("None of the requested columns were found in the dataset: " + ", ".join(requested_cols))
    else:
        df_work = df_corr_src[cols].copy()
        for col in df_work.columns:
            if df_work[col].dtype == 'object' or df_work[col].dtype.name == 'category':
                df_work[col] = pd.factorize(df_work[col].astype(str))[0]

        corr_matrix = df_work.corr()

        colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
        cmap = LinearSegmentedColormap.from_list("green_palette", colors_palette, N=256)

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

    # ---------------- Genetics stacked bar & insights ----------------
    try:
        changeWithTarget = df_predict_cleaned.groupby(['Genetic_Encoding', 'Hair_Loss']).size().unstack(fill_value=0)

        # Calculate proportions safely (handle if index values missing)
        if 1 in changeWithTarget.index:
            prop_with_genetic = changeWithTarget.loc[1] / changeWithTarget.loc[1].sum() * 100
        else:
            prop_with_genetic = pd.Series(dtype=float)
        if 0 in changeWithTarget.index:
            prop_without_genetic = changeWithTarget.loc[0] / changeWithTarget.loc[0].sum() * 100
        else:
            prop_without_genetic = pd.Series(dtype=float)
        
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

        fig, ax = plt.subplots(figsize=(10,6))
        changeWithTarget.plot(
            kind='bar',
            stacked=True,
            alpha=0.9,
            ax=ax,
            color = ["#bddb8e", "#85ada6", "#A8D5BA", "#5d9189", "#2E8B57"]
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
    except Exception as e:
        st.error("Graph cannot be created.")
        print("Error creating genetics plots:", e)

    # ---------------- Nutritional deficiencies vs hair loss (genetic subset) ----------------
    try:
        df_genetic = df_predict_cleaned[df_predict_cleaned['Genetic_Encoding'] == 1]
        counts = df_genetic.groupby(['Nutritional_Deficiencies', 'Hair_Loss']).size().reset_index(name='Count')
        counts['Hair_Loss'] = counts['Hair_Loss'].astype(str)

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
        fig, ax = plt.subplots(figsize=(14,6))
        sns.barplot(
            data=counts,
            x='Nutritional_Deficiencies',
            y='Count',
            hue='Hair_Loss',
            ax=ax,
            palette=green_palette
        )
        ax.set_title('Hair Loss Distribution by Nutritional Deficiency (Genetic Hair Loss = 1)', fontsize=16, pad=15)
        ax.set_xlabel('Nutritional Deficiencies', fontsize=14, labelpad=10)
        ax.set_ylabel('Number of People', fontsize=14, labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
        ax.legend(title='Hair Loss', labels=['No (0)', 'Yes (1)'], fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("Graph cannot be created.")
        print("Error creating nutritional deficiency plot:", e)

    # ---------------- Alopecia / Age trends ----------------
    try:
        bins = [18, 25, 40, 51]
        labels = ['18-25', '25-40', '40-51']
        df_predict_cleaned['Age_Range_New'] = pd.cut(df_predict_cleaned['Age'], bins=bins, labels=labels, right=False)

        alopecia_conditions = ['Alopecia Areata', 'Androgenetic Alopecia']
        df_alopecia = df_predict_cleaned[df_predict_cleaned['Medical_Conditions'].isin(alopecia_conditions)]

        alopecia_counts = df_alopecia.groupby(['Age_Range_New', 'Medical_Conditions'])['Hair_Loss'].sum().unstack(fill_value=0)

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

        # filter and plot for genetic + alopecia
        df_genetic = df_predict_cleaned[(df_predict_cleaned['Genetic_Encoding'] == 1) & 
                            (df_predict_cleaned['Medical_Conditions'].isin(alopecia_conditions))]

        counts = df_genetic.groupby(['Medical_Conditions', 'Hair_Loss']).size().reset_index(name='Count')
        counts['Hair_Loss'] = counts['Hair_Loss'].astype(str)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=counts, x='Medical_Conditions', y='Count', hue='Hair_Loss', palette=["#a4d4be", "#85ada6"], ax=ax)
        ax.set_title('Hair Loss by Alopecia Type (Genetic Encoding = 1)', fontsize=9, pad=15)
        ax.set_xlabel('Alopecia Type', fontsize=9)
        ax.set_ylabel('Number of People', fontsize=9)
        ax.legend(title='Hair Loss', labels=['No (0)', 'Yes (1)'], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

        # plot alopecia_counts by age range
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

    except Exception as e:
        st.error("Graph cannot be created.")
        print("Error creating alopecia/age trend plots:", e)
