# /EDA/Nutrition_Dataset.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Local copies of color/style tokens used across other EDA modules
BASE_BG = "#FFFFFF"
ACCENT = "#FFFFFF"
HEADER_COLOR = "#2E8B57"
TEXT = "#2C3E50"
SECTION_BG_PLOTS = "#749683"
CARD_COLOR = "#d4e6e4"

def render_nutrition_page(df):
    
    st.markdown(f"""
    <div padding:14px; border-radius:12px;'>
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
       In this section we explore the nutritional properties of foods in the dataset:
            which foods have the highest nutrient density, how foods distribute across different nutritional categories,
            and quick visual comparisons to help prioritize nutrient-rich options.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f""" <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Top Foods by Nutritional Metric</h3> </div> """, unsafe_allow_html=True) 
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; margin-top:8px;'>
    Below you can choose a nutritional metric and a category to filter by — the chart will show the top <b>n</b> foods for that selection.
    By default we display the top 20 foods by <b>Nutrition Density</b>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.error("Nutrition dataset not loaded.")
        return

    data = df.copy()

    # --- detect sensible columns ---
    # Food column (exact match fallback)
    food_col_candidates = [c for c in data.columns if c.lower() in ('food', 'food_name', 'item')]
    food_col = food_col_candidates[0] if food_col_candidates else (data.columns[0] if len(data.columns) > 0 else None)

    # Metric columns (numeric). Prefer 'Nutrition_Density' if present.
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    preferred_metric = 'Nutrition_Density'
    metric_col = preferred_metric if preferred_metric in data.columns else (numeric_cols[0] if numeric_cols else None)

    # Categorical columns for grouping (exclude 'Food' and numeric)
    cat_cols = [c for c in data.select_dtypes(include=['object', 'category']).columns if c != food_col]
    # Provide fallback: small-cardinality numeric columns could also represent categories (encoded)
    for c in data.columns:
        if c not in cat_cols and c != metric_col and c != food_col:
            # treat as categorical if few unique values
            try:
                if data[c].nunique(dropna=True) <= 30:
                    cat_cols.append(c)
            except Exception:
                pass



    # --- UI controls ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if metric_col:
            metric_choice = st.selectbox("Select metric (y-axis):", options=[metric_col] + [c for c in numeric_cols if c != metric_col], index=0)
        else:
            metric_choice = st.selectbox("Select metric (y-axis):", options=["(no numeric columns found)"], index=0)

    with col2:
        if cat_cols:
            category_choice = st.selectbox("Select category (optional):", options=["All"] + cat_cols, index=0)
        else:
            category_choice = "All"

    with col3:
        top_n = st.number_input("Top N", min_value=1, max_value=200, value=20, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Data preparation ---
    try:
        # If user selected a category, filter dataset to that category
        working = data.copy()
        if category_choice != "All" and category_choice in working.columns:
            # allow user to pick a specific category value optionally
            unique_vals = working[category_choice].dropna().unique().tolist()
            unique_vals_sorted = sorted(unique_vals, key=lambda x: str(x))
            selected_val = st.selectbox(f"Filter {category_choice} by value (optional):", options=["All"] + unique_vals_sorted, index=0)
            if selected_val != "All":
                working = working[working[category_choice] == selected_val].copy()

        # Clean metric & food columns
        if metric_choice not in working.columns:
            st.error(f"Metric column '{metric_choice}' not found in data.")
            return
        if food_col not in working.columns:
            st.error("Food column not found in data.")
            return

        # Drop NA in required cols and ensure numeric metric
        tmp = working[[food_col, metric_choice]].dropna(subset=[metric_choice, food_col]).copy()
        tmp[metric_choice] = pd.to_numeric(tmp[metric_choice], errors='coerce')
        tmp = tmp.dropna(subset=[metric_choice])

        if tmp.empty:
            st.info("No data available for the selected filters.")
            return

        top_n = int(top_n)
        top = tmp.sort_values(by=metric_choice, ascending=False).head(top_n)

        # --- Plot ---
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10,6))  # height scales with top_n
        # Capitalize food names
        food_names = top[food_col].astype(str).str.title()[::-1]
        ax.barh(food_names, top[metric_choice].values[::-1], color="#77a48f")
        ax.set_xlabel(f'{metric_choice} (g)')
        ax.set_title(f'Top {top_n} Foods by {metric_choice}')
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

       # --- Show table and allow CSV download ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<b>Top {top_n} foods (table):</b>", unsafe_allow_html=True)

        # Prepare display copy
        top_display = top.copy()
        # Capitalize food names
        top_display[food_col] = top_display[food_col].astype(str).str.title()
        # Round metric to whole numbers
        top_display[metric_choice] = top_display[metric_choice].round(0).astype(int)

        st.dataframe(top_display.reset_index(drop=True))

        # CSV download
        csv = top_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download top foods as CSV",
            data=csv,
            file_name=f"top_{top_n}_foods_by_{metric_choice}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error("Could not compute or plot top foods.")
        print("Nutrition page error:", e)

    # --- Section 2: Macronutrient Composition for Top Foods ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> 
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Macronutrient Composition — Top Foods</h3> 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
    This section visualizes the macronutrient composition (Fat, Carbohydrates, Protein) of top foods. 
    You can quickly see which foods are protein-rich, carb-heavy, or high in fat. 
    By default, the chart shows the top 15 foods ranked by Protein content, but you can select other macronutrients to sort by.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- UI Controls ---
    macro_cols = ['Protein', 'Carbohydrates', 'Fat']
    macro_choice = st.selectbox("Sort top foods by:", options=macro_cols, index=0)  # default Protein
    top_n_macro = st.number_input("Top N foods for macro chart:", min_value=5, max_value=50, value=15, step=1)

        # --- Prepare Data ---
    try:
        top_macro = data[['Food'] + macro_cols].dropna(subset=macro_cols)
        top_macro = top_macro.sort_values(by=macro_choice, ascending=False).head(top_n_macro)
        top_macro['Food'] = top_macro['Food'].astype(str).str.title()  # Capitalize food names
        top_macro = top_macro.set_index('Food')

        # --- Plot ---
        plt.figure(figsize=(12,6))
        bottom = pd.Series([0]*len(top_macro), index=top_macro.index)
        colors = ["#85ada6", "#E67E22", "#909492"]  # Protein, Carbs, Fat

        for i, c in enumerate(macro_cols):
            plt.bar(top_macro.index, top_macro[c], bottom=bottom, label=c, color=colors[i])
            bottom += top_macro[c]

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('g per serving (or per 100g)')
        plt.title(f'Stacked Macro Composition for Top {top_n_macro} Foods (by {macro_choice})')
        plt.legend()
        plt.tight_layout()

        st.pyplot(plt)
        plt.close()

    except Exception as e:
        st.error("Could not generate macronutrient stacked bar chart.")
        print("Macro chart error:", e)

    # --- Section 3: Vitamin Radar Chart for a Single Food ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> 
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Vitamin Profile — Single Food</h3> 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
    This section shows a radar chart representing the vitamin profile of a single food. 
    By default, <b>Cream Cheese</b> is displayed, but you can select any other food from the dataset to visualize its vitamin content.
    The chart normalizes values for better comparison across vitamins.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- UI Control ---
    food_options = data['Food'].dropna().astype(str).str.title().unique().tolist()
    food_options.sort() 
    default_food = 'Cream Cheese' if 'Cream Cheese' in food_options else food_options[0]
    food_choice = st.selectbox("Select food for vitamin profile:", options=food_options, index=food_options.index(default_food))

    # --- Prepare Data ---
    vit_list = ['Vitamin_A','Vitamin_B1','Vitamin_B11','Vitamin_B12','Vitamin_B2',
                'Vitamin_B3','Vitamin_B5','Vitamin_B6','Vitamin_C','Vitamin_D','Vitamin_E','Vitamin_K']

    try:
        row = data[data['Food'].str.title() == food_choice].iloc[0]
        values = row[vit_list].fillna(0).values
        max_val = max(values.max(), 1)
        values = values / max_val  # normalize

        angles = np.linspace(0, 2 * np.pi, len(vit_list), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += [angles[0]]

        # --- Plot ---
        plt.figure(figsize=(8,8))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color="#77a48f")  # green line
        ax.fill(angles, values, alpha=0.25, color="#77a48f")  # matching green fill
        ax.set_thetagrids(np.degrees(angles[:-1]), vit_list)
        ax.set_title(f'Vitamin Profile (normalized) — {food_choice}', y=1.1)
        plt.tight_layout()

        st.pyplot(plt)
        plt.close()

    except Exception as e:
        st.error("Could not generate vitamin radar chart for the selected food.")
        print("Vitamin radar error:", e)
        
        # --- Section: Scatter Plot of Nutrients ---
    st.markdown(f""" 
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:30px;'> 
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Scatter Plot of Nutrients</h3> 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
    This section allows you to explore relationships between two nutrient metrics across foods. 
    By default, we show <b>Carbohydrates vs Sugars</b>, but you can select any numeric variables from the dataset for comparison. 
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Numeric columns for selection ---
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # --- User selection for X and Y ---
    col1, col2 = st.columns(2)
    with col1:
        x_choice = st.selectbox("X-axis:", options=numeric_cols, index=numeric_cols.index('Carbohydrates') if 'Carbohydrates' in numeric_cols else 0)
    with col2:
        y_choice = st.selectbox("Y-axis:", options=numeric_cols, index=numeric_cols.index('Sugars') if 'Sugars' in numeric_cols else 0)

    # --- Scatter plot ---
    try:
        plt.figure(figsize=(8,6))
        
        # Use green gradient for point color
        x = data[x_choice]
        y = data[y_choice]
        # normalize color intensity based on distance from origin
        distances = np.sqrt((x - x.min())**2 + (y - y.min())**2)
        norm = (distances - distances.min()) / (distances.max() - distances.min())
        colors = plt.cm.Greens(norm*0.4 + 0.6)  # start with lighter shade (#c1dab8) and darker as values increase

        plt.scatter(x, y, alpha=0.7, color=colors, edgecolor='k', s=25)

        plt.xlabel(f'{x_choice} (g)')
        plt.ylabel(f'{y_choice} (g)')
        plt.title(f'{x_choice} vs {y_choice}')

        # annotate top 6 points by y_choice
        for _, r in data.nlargest(6, y_choice)[['Food', x_choice, y_choice]].iterrows():
            plt.annotate(str(r['Food']).title(), (r[x_choice], r[y_choice]), textcoords="offset points", xytext=(5,5))

        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    except Exception as e:
        st.error("Could not generate scatter plot.")
        print("Scatter plot error:", e)
    
        # --- Section: Compare Two Foods (Side-by-Side Summary) ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:{SECTION_BG_PLOTS}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'> 
        <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Compare Two Foods — Side by Side</h3> 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; margin:0;'>
    Select two foods to compare their nutrient profiles. By default the first two foods (alphabetically) are selected.
    You will see a compact table comparing key metrics, a stacked macronutrient bar chart for each food, and vitamin radar charts side-by-side.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Ensure data exists
    if df is None:
        st.info("Nutrition data not available for comparison.")
    else:
        # Normalize food names and get sorted unique list
        data['Food_Title'] = data[food_col].astype(str).str.title()
        food_options = sorted(data['Food_Title'].dropna().unique().tolist())

        if len(food_options) < 2:
            st.info("Need at least two foods in the dataset to compare.")
        else:
            # default selections: first two alphabetically
            default_a = food_options[0]
            default_b = food_options[1] if len(food_options) > 1 else food_options[0]

            col_a, col_b = st.columns(2)
            with col_a:
                food_a = st.selectbox("Food A:", options=food_options, index=food_options.index(default_a))
            with col_b:
                # ensure second list doesn't pick the same by default (but allow user to pick same)
                remaining = food_options.copy()
                # keep ordering stable
                idx_b = remaining.index(default_b) if default_b in remaining else 0
                food_b = st.selectbox("Food B:", options=remaining, index=idx_b)

            # fetch rows (case-insensitive title match)
            row_a = data[data['Food_Title'] == food_a]
            row_b = data[data['Food_Title'] == food_b]

            if row_a.empty or row_b.empty:
                st.error("Could not find one or both foods in the data after normalization.")
            else:
                row_a = row_a.iloc[0]
                row_b = row_b.iloc[0]

                # KEY METRICS to show in table (try to pick sensible columns)
                # Prefer some common nutrition columns if present
                preferred_metrics = [
                    'Calories', 'Protein', 'Fat', 'Carbohydrates', 'Sugars', 'Fiber',
                    'Sodium', 'Calcium', 'Iron', 'Vitamin_A', 'Vitamin_C', 'Vitamin_D', 'Nutrition_Density'
                ]
                metrics = [m for m in preferred_metrics if m in data.columns]
                # fallback: pick top 8 numeric cols
                if not metrics:
                    metrics = data.select_dtypes(include=[np.number]).columns.tolist()[:8]

                # Prepare comparison table
                comp = pd.DataFrame({
                    'Metric': metrics,
                    f'{food_a}': [row_a.get(m, np.nan) for m in metrics],
                    f'{food_b}': [row_b.get(m, np.nan) for m in metrics]
                })
                # Round numeric metrics for display (Nutrition_Density as whole number)
                for c in [f'{food_a}', f'{food_b}']:
                    comp[c] = pd.to_numeric(comp[c], errors='coerce')
                if 'Nutrition_Density' in metrics:
                    comp.loc[comp['Metric'] == 'Nutrition_Density', f'{food_a}'] = comp.loc[comp['Metric'] == 'Nutrition_Density', f'{food_a}'].round(0).astype('Int64')
                    comp.loc[comp['Metric'] == 'Nutrition_Density', f'{food_b}'] = comp.loc[comp['Metric'] == 'Nutrition_Density', f'{food_b}'].round(0).astype('Int64')

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Quick comparison table:**", unsafe_allow_html=True)
                st.dataframe(comp.set_index('Metric'))

                # --- Macronutrient stacked bars (Protein, Carbohydrates, Fat) for each food ---
                macro_cols = [c for c in ['Protein', 'Carbohydrates', 'Fat'] if c in data.columns]
                if macro_cols:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Macronutrient breakdown (g)**", unsafe_allow_html=True)

                    fig, axes = plt.subplots(ncols=2, figsize=(12,4), sharey=True)
                    # colors: Protein, Carbs, Fat -> use requested palette ["#85ada6", "#E67E22", "#D9D9D9"]
                    palette = ["#85ada6", "#E67E22", "#909492"]
                    for ax, r, title in zip(axes, [row_a, row_b], [food_a, food_b]):
                        values = [float(r.get(c, 0) or 0) for c in macro_cols]
                        bottoms = np.zeros(len(values))
                        # Single horizontal stacked bar (vertical orientation is easier to read in two columns)
                        ax.barh([0]*len(values), values, left=[sum(values[:i]) for i in range(len(values))],
                                color=[palette[i % len(palette)] for i in range(len(values))])
                        # create a legend and label
                        ax.set_title(title)
                        ax.set_yticks([])
                        # annotate numeric values on bar segments
                        total = sum(values)
                        if total > 0:
                            cum = 0
                            for i, v in enumerate(values):
                                if v > 0:
                                    ax.text(cum + v/2, 0, f"{macro_cols[i]}: {v:.0f}", va='center', ha='center', fontsize=9, color='black')
                                cum += v
                        ax.set_xlim(0, max(1, total*1.15))
                    plt.suptitle("Macronutrient composition")
                    plt.tight_layout(rect=[0,0,1,0.95])
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Macronutrient columns not available for stacked bar comparison.")

                # --- Vitamins radar charts side-by-side ---
                vit_list = [v for v in ['Vitamin_A','Vitamin_B1','Vitamin_B11','Vitamin_B12','Vitamin_B2',
                                        'Vitamin_B3','Vitamin_B5','Vitamin_B6','Vitamin_C','Vitamin_D','Vitamin_E','Vitamin_K']
                            if v in data.columns]
                if vit_list:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Vitamin profile (normalized)**", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    try:
                        # Prepare normalized values function
                        def get_normed_vals(row):
                            vals = row[vit_list].fillna(0).astype(float).values
                            max_val = max(vals.max(), 1)
                            vals = vals / max_val
                            vals = np.concatenate((vals, [vals[0]]))
                            return vals

                        angles = np.linspace(0, 2 * np.pi, len(vit_list), endpoint=False).tolist()
                        angles += [angles[0]]

                        with col1:
                            vals_a = get_normed_vals(row_a)
                            plt.figure(figsize=(5,5))
                            ax = plt.subplot(111, polar=True)
                            ax.plot(angles, vals_a, 'o-', linewidth=2, color="#77a48f")
                            ax.fill(angles, vals_a, alpha=0.25, color="#77a48f")
                            ax.set_thetagrids(np.degrees(angles[:-1]), vit_list)
                            ax.set_title(food_a, fontsize=10, y=1.08)
                            plt.tight_layout()
                            st.pyplot(plt)
                            plt.close()

                        with col2:
                            vals_b = get_normed_vals(row_b)
                            plt.figure(figsize=(5,5))
                            ax = plt.subplot(111, polar=True)
                            ax.plot(angles, vals_b, 'o-', linewidth=2, color="#77a48f")
                            ax.fill(angles, vals_b, alpha=0.25, color="#77a48f")
                            ax.set_thetagrids(np.degrees(angles[:-1]), vit_list)
                            ax.set_title(food_b, fontsize=10, y=1.08)
                            plt.tight_layout()
                            st.pyplot(plt)
                            plt.close()

                    except Exception as e:
                        st.error("Could not generate vitamin radar charts.")
                        print("Radar compare error:", e)
                else:
                    st.info("Vitamin columns not available for radar comparison.")

                # --- Download combined comparison as CSV ---
                try:
                    comp_csv = comp.copy()
                    # convert to a clean CSV-friendly format
                    comp_csv = comp_csv.set_index('Metric').T.reset_index().rename(columns={'index':'Food'})
                    comp_csv.loc[comp_csv['Food'] == food_a, 'Food'] = food_a
                    comp_csv.loc[comp_csv['Food'] == food_b, 'Food'] = food_b
                    csv_bytes = comp_csv.to_csv(index=False).encode('utf-8')
                    st.download_button("Download comparison CSV", data=csv_bytes, file_name=f"compare_{food_a.replace(' ','_')}_vs_{food_b.replace(' ','_')}.csv", mime="text/csv")
                except Exception:
                    pass
