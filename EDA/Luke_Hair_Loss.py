# /EDA/Luke_Hair_Loss.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder

# Local copies of the style/colors used by the main page
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

def render_luke_page(df_luke_cleaned):
    """
    Render the Luke Hair Loss EDA page.
    Expects df_luke_cleaned as a cleaned pandas DataFrame (or None).
    """

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

    if df_luke_cleaned is None:
        st.error("Luke Hair Dataset not loaded.")
        return
    df = df_luke_cleaned.copy()

    # Ensure default column exists
    default_col = 'Hair_Loss' if 'Hair_Loss' in df.columns else (df.columns[0] if len(df.columns)>0 else None)

    col_options = [
        'Date', 'Hair_Loss', 'Stay_Up_Late', 'Pressure_Level', 
        'Coffee_Consumed', 'Brain_Working_Duration', 'School_Assesssment', 
        'Stress_Level', 'Shampoo_Brand', 'Swimming', 'Hair_Washing', 
        'Hair_Grease', 'Dandruff', 'Libido'
    ]
    # Filter options to existing columns only
    col_options = [c for c in col_options if c in df.columns]

    if not col_options:
        st.error("No expected columns found in Luke dataset.")
        return

    # default index
    try:
        default_index = col_options.index('Hair_Loss') if 'Hair_Loss' in col_options else 0
    except ValueError:
        default_index = 0

    col_choice = st.selectbox("Feature:", options=col_options, index=default_index)

    # Define custom orderings used for nicer plots
    custom_orders = {
        'Hair_Loss': ['Few', 'Medium', 'Many', 'A lot'],
        'Pressure_Level': ['Low', 'Medium', 'High', 'Very High'],
        'Stress_Level': ['Low', 'Medium', 'High', 'Very High'],
        'Dandruff': ['None', 'Few', 'Many'],
        'Hair_Washing': ['No', 'Yes'],
        'Swimming': ['No', 'Yes'],
        'School_Assesssment': ['No assessment', 'Individual Assessment', 'Team Assessment', 'Final exam', 'Final exam revision' ]
    }

    # Distribution plot
    try:
        col_data = df[col_choice].fillna('Missing').astype(str)
        if col_choice in custom_orders:
            categories = custom_orders[col_choice]
            counts = col_data.value_counts().reindex(categories, fill_value=0)
        elif col_choice in ['Coffee_Consumed', 'Brain_Working_Duration', 'Stay_Up_Late']:
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            counts = col_numeric.value_counts().sort_index()
            categories = counts.index.tolist()
        else:
            counts = col_data.value_counts().sort_index()
            categories = counts.index.tolist()

        values = counts.values

        colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
        colors = [colors_palette[i % len(colors_palette)] for i in range(len(categories))]

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
            height=650,
            plot_bgcolor="white",
            paper_bgcolor="white",
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

    # Relationship explorer
    # restrict selectable columns to those present
    cols_list = df.columns.tolist()
    default_x = 'Stay_Up_Late' if 'Stay_Up_Late' in cols_list else cols_list[0]
    default_y = 'Hair_Loss' if 'Hair_Loss' in cols_list else (cols_list[0] if len(cols_list)>0 else None)

    x_choice = st.selectbox("X-axis:", options=cols_list, index=cols_list.index(default_x))
    y_choice = st.selectbox("Y-axis:", options=cols_list, index=cols_list.index(default_y))

    try:
        df_plot = df[[x_choice, y_choice]].copy()
        df_plot[y_choice] = df_plot[y_choice].fillna('Missing').astype(str)
        df_plot[x_choice] = df_plot[x_choice].fillna('Missing')

        # categories for x
        if x_choice in custom_orders:
            categories_x = custom_orders[x_choice]
        elif x_choice in ['Coffee_Consumed', 'Brain_Working_Duration', 'Stay_Up_Late']:
            col_numeric = pd.to_numeric(df_plot[x_choice], errors='coerce')
            counts_x = col_numeric.value_counts().sort_index()
            categories_x = counts_x.index.tolist()
        else:
            categories_x = df_plot[x_choice].astype(str).value_counts().sort_index().index.tolist()

        # categories for y
        if y_choice in custom_orders:
            categories_y = custom_orders[y_choice]
        else:
            categories_y = df_plot[y_choice].astype(str).unique().tolist()

        counts = df_plot.groupby([x_choice, y_choice]).size().reset_index(name='Count')

        colors_palette = ["#b7e4c7", "#2db17a", "#186b46", "#d9a044", "#74c69d", "#c744c1"]

        fig = go.Figure()
        for i, cat in enumerate(categories_y):
            cat_data = counts[counts[y_choice] == cat]
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
        st.plotly_chart(fig, use_container_width=True, config=config)

    except Exception as e:
        st.error("Graph cannot be created.")
        print(f"Error creating interactive 2-variable plot: {e}")

    # Correlation heatmap
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

    # Choose a safe set of columns present in dataframe for correlation
    candidate_cols = ['Date', 'Hair_Loss_Encoding', 'Stay_Up_Late', 'Pressure_Level_Encoding', 
        'Coffee_Consumed', 'Brain_Working_Duration', 'School_Assesssment', 
        'Stress_Level_Encoding', 'Shampoo_Brand', 'Swimming', 'Hair_Washing', 
        'Hair_Grease', 'Dandruff_Encoding', 'Libido']

    cols = [c for c in candidate_cols if c in df.columns]
    if cols:
        try:
            df_corr = df[cols].copy()
            for col in df_corr.select_dtypes(include='object').columns:
                df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))

            corr_matrix = df_corr.corr()

            colors_palette = ["#c1dab8", "#77a48f", "#4e8f73", "#407059", "#255B42", "#0E3A26"]
            cmap = LinearSegmentedColormap.from_list("green_palette", colors_palette, N=256)

            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True, linewidths=0.8, linecolor='white')
            plt.title("Correlation Heatmap — Luke Dataset", fontsize=16, color=HEADER_COLOR)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error("Could not create correlation heatmap.")
            print("Correlation heatmap error:", e)
    else:
        st.info("No suitable columns found for correlation heatmap in Luke dataset.")

    # Hair loss over time
    st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Hair Loss Over Time (Smoothened)</h3> </div> """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
            <p style='font-size:18px; color:{TEXT}; margin:0;'>
            Using this time series plot, we can observe that there is no consistent trend in hair fall over time. 
            Any increases or decreases appear to depend on other factors such as <b>Coffee Consumption</b>, <b>Staying Up Late</b>, or <b>Stress Levels</b>.
            </p>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if 'Date' in df.columns and 'Hair_Loss_Encoding' in df.columns:
        try:
            df_time = df.copy()
            # try common date formats; if fails user will see error printed
            df_time['Date'] = pd.to_datetime(df_time['Date'], errors='coerce', dayfirst=True)
            df_time = df_time.sort_values('Date').dropna(subset=['Date']).copy()
            if df_time.empty:
                st.info("No valid date values to plot time series.")
            else:
                df_time['Smoothed_Hair_Loss'] = df_time['Hair_Loss_Encoding'].rolling(window=5, center=True, min_periods=1).mean()
                fig = px.line(df_time, x='Date', y='Smoothed_Hair_Loss', labels={'Smoothed_Hair_Loss': 'Hair Loss (Smoothed)', 'Date':'Date'}, line_shape='spline', template='simple_white')
                fig.update_traces(line=dict(color='mediumseagreen', width=3), hovertemplate='Date: %{x|%d-%b-%Y}<br>Hair Loss: %{y:.2f}<extra></extra>')
                fig.update_layout(height=500, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Could not create time series plot.")
            print("Time series error:", e)
    else:
        st.info("Required columns for time series ('Date' and 'Hair_Loss_Encoding') not present.")

    # Insight summary
    st.markdown(f""" <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Useful Insights </h3> </div> """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        The analysis reveals that <b>Hair Loss</b> has a positive correlation with several lifestyle and behavioral factors such as 
        <b>Staying Up Late</b>, <b>Long Brain Working Duration</b>, <b>Hair Grease</b>, <b>Coffee Consumed</b>, <b>Pressure Level</b>, 
        and <b>Stress Level</b>. 
        <br><br>
        This means that whenever Luke experienced higher stress or pressure levels, worked for extended periods, consumed more coffee, 
        or frequently stayed up late, the likelihood of hair loss tends to increase. 
        Similarly, higher hair grease buildup — which may block follicles — also appears linked to increased hair shedding. 
        Overall, these findings suggest that both <b>lifestyle habits</b> and <b>mental well-being</b> play a significant role in influencing hair health.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Correlations with Hair_Loss_Encoding (horizontal bar)
    corr_cols = ['Hair_Loss_Encoding', 'Stay_Up_Late', 'Pressure_Level_Encoding', 'Coffee_Consumed', 
                 'Brain_Working_Duration', 'Stress_Level_Encoding', 'Shampoo_Brand', 'Swimming', 'Hair_Washing', 
                 'Hair_Grease', 'Dandruff_Encoding', 'Libido']
    corr_cols = [c for c in corr_cols if c in df.columns]
    if 'Hair_Loss_Encoding' in corr_cols:
        try:
            df_corr = df[corr_cols].copy()
            for col in df_corr.select_dtypes(include='object').columns:
                df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
            corr_matrix = df_corr.corr()
            if 'Hair_Loss_Encoding' in corr_matrix.columns:
                hair_loss_corr = corr_matrix['Hair_Loss_Encoding'].drop('Hair_Loss_Encoding').sort_values(ascending=False)
                if not hair_loss_corr.empty:
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
                    fig.update_layout(title=dict(text="Feature Correlation with Hair_Loss", x=0.35, font=dict(size=22)), xaxis=dict(range=[-1,1]), height=600, template="simple_white")
                    fig.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>Correlation: %{x:.2f}<extra></extra>')
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Could not compute correlations with Hair_Loss.")
            print("Hair loss correlation error:", e)
    else:
        st.info("Hair_Loss_Encoding not present — cannot compute feature correlations with Hair_Loss.")

    # Bubble charts and 3D scatter sections
    # Provide interactive bubble chart selection (safe-guard columns exist)
    available_vars = [c for c in ['Stay_Up_Late','Coffee_Consumed','Brain_Working_Duration','Stress_Level','Pressure_Level','Hair_Grease','Dandruff','Libido'] if c in df.columns]
    enc_vars = [c for c in ['Stress_Level_Encoding','Pressure_Level_Encoding','Dandruff_Encoding','Hair_Loss_Encoding'] if c in df.columns]

    if available_vars and 'Hair_Loss_Encoding' in df.columns:
        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:40px;'> 
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Bubble Chart of Hair Loss vs Second Variable with Third Variable as Size & Color</h3> 
        </div> 
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        This interactive <b>Bubble Chart</b> helps explore how <b>two lifestyle or behavioral factors</b> jointly influence hair loss. 
        Each point represents an observation, where:
        <br>• The <b>x-axis</b> shows the chosen second variable,  
        <br>• The <b>y-axis</b> represents the <b>Hair Loss Encoding</b> (severity of hair loss),  
        <br>• The <b>bubble size and color</b> correspond to a third variable of your choice.  
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        second_var = st.selectbox("Select the second variable (X-axis):", options=available_vars, index=0)
        third_var_candidates = [v for v in available_vars + enc_vars if v in df.columns and v!= second_var]
        if not third_var_candidates:
            st.info("No suitable third variable for bubble chart available.")
        else:
            third_var = st.selectbox("Select the third variable (Bubble Size & Color):", options=third_var_candidates, index=0)
            try:
                fig3 = px.scatter(
                    df,
                    x=second_var,
                    y='Hair_Loss_Encoding',
                    size=third_var if third_var in df.columns else None,
                    color=third_var if third_var in df.columns else None,
                    color_continuous_scale='Viridis',
                    hover_data=[c for c in ['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'] if c in df.columns],
                    title=f'Bubble Chart of Hair Loss vs {second_var} with {third_var} as Size & Color'
                )
                fig3.update_layout(template="simple_white", title_x=0.05, title_font=dict(size=20), xaxis_title=second_var, yaxis_title="Hair Loss Encoding", hovermode="closest", height=700)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error("Could not generate the bubble chart.")
                print(f"Error generating bubble chart: {e}")

        # 4-variable bubble chart (size + color)
        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:40px;'> 
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Bubble Chart of Hair Loss with Multi-Variable Interaction</h3> 
        </div> 
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        second_var_4 = st.selectbox("Select X variable (4-var bubble):", options=available_vars, index=0, key="bubble_x")
        third_var_4 = st.selectbox("Select Bubble Size:", options=[v for v in available_vars+enc_vars if v in df.columns and v!=second_var_4], index=0, key="bubble_size")
        fourth_var_4 = st.selectbox("Select Bubble Color:", options=[v for v in available_vars+enc_vars if v in df.columns and v not in [second_var_4, third_var_4]], index=0, key="bubble_color")
        try:
            fig4 = px.scatter(
                df,
                x=second_var_4,
                y='Hair_Loss_Encoding',
                size=third_var_4 if third_var_4 in df.columns else None,
                color=fourth_var_4 if fourth_var_4 in df.columns else None,
                color_continuous_scale='Viridis',
                hover_data=[c for c in ['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'] if c in df.columns],
                title=f'Bubble Chart of Hair Loss vs {second_var_4} (Size = {third_var_4}, Color = {fourth_var_4})'
            )
            fig4.update_layout(template="simple_white", title_font=dict(size=22), xaxis_title=second_var_4, yaxis_title="Hair Loss Encoding", height=750)
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.error("Could not generate the 4-variable bubble chart.")
            print(f"Error generating 4-var bubble chart: {e}")

        # 3D scatter (Section A)
        st.markdown(f""" 
        <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:30px;'> 
            <h3 style='color:{ACCENT}; font-size:22px; margin:6px 0;'>3D Scatter: Hair Loss vs Second Variable (Size & Color = Third Variable)</h3> 
        </div> 
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        second_var_a = st.selectbox("Section A — Select X variable", options=available_vars, index=0, key="secA_x")
        third_var_a = st.selectbox("Section A — Select Z / Size / Color variable", options=[v for v in available_vars+enc_vars if v in df.columns and v!=second_var_a], index=0, key="secA_z")
        try:
            needed_cols = [c for c in [second_var_a, 'Hair_Loss_Encoding', third_var_a, 'Hair_Loss'] if c in df.columns]
            tmp = df[needed_cols].copy()
            for col in [second_var_a, 'Hair_Loss_Encoding', third_var_a]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')

            # drop NA on core axes
            core_axes = [c for c in [second_var_a, 'Hair_Loss_Encoding', third_var_a] if c in tmp.columns]
            tmp = tmp.dropna(subset=core_axes)

            hover_cols = [c for c in ['Hair_Loss', 'Brain_Working_Duration', 'Stress_Level'] if c in tmp.columns]
            if tmp.empty:
                st.info("No data available after filtering for 3D scatter.")
            else:
                fig_a = px.scatter_3d(
                    tmp,
                    x=second_var_a if second_var_a in tmp.columns else None,
                    y='Hair_Loss_Encoding' if 'Hair_Loss_Encoding' in tmp.columns else None,
                    z=third_var_a if third_var_a in tmp.columns else None,
                    size=third_var_a if third_var_a in tmp.columns else None,
                    color=third_var_a if third_var_a in tmp.columns else None,
                    color_continuous_scale='Viridis',
                    hover_data=hover_cols,
                    title=f'\n\nHair Loss vs {second_var_a} vs {third_var_a} (size & color = {third_var_a})'
                )
                fig_a.update_layout(height=700, scene=dict(xaxis_title=second_var_a, yaxis_title='Hair_Loss_Encoding', zaxis_title=third_var_a), margin=dict(l=20, r=20, t=70, b=20))
                st.plotly_chart(fig_a, use_container_width=True)
        except Exception as e:
            st.error("Could not generate Section A 3D chart. See console for details.")
            print("Section A error:", e)
    else:
        st.info("Not enough variables available for bubble/3D visualizations in this dataset.")
