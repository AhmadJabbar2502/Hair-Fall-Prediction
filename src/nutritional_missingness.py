# src/nutritional_missingness.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

sns.set_style("whitegrid")

def impute_nutritional_mode(row, df):
    """
    Fill missing Nutritional_Deficiencies values with the mode within the same Age_Range.
    If no mode exists for that group, return 'Unknown'.
    """
    if pd.isna(row['Nutritional_Deficiencies']):
        group = df[df['Age_Range'] == row['Age_Range']]
        mode_val = group['Nutritional_Deficiencies'].mode()
        if not mode_val.empty:
            return mode_val[0]
        else:
            return 'Unknown'
    else:
        return row['Nutritional_Deficiencies']

def render_nutritional_missingness(hair_raw, df_raw, df_cleaned,
                                  HEADER_COLOR="#2E8B57", TEXT="#333",
                                  SECTION_BG="#F0F8F5", ACCENT="#2E8B57"):
    """
    Renders the Nutritional_Deficiencies missingness + imputation comparisons in Streamlit.
    Expects hair_raw (for missingness analysis), df_raw (before imputation), df_cleaned (after RF imputation).
    """

    st.markdown(f"""
        <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingness in Nutritional Deficiencies</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Observation card
    st.markdown(f"""
    <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        <b>Observation:</b> Missingness in <b>Nutritional_Deficiencies</b> is not completely random (not MCAR). Chi-squared tests reveal a strong dependency on <b>Age Range</b>. Most missing entries are concentrated in younger ages (e.g., 18-30), so these values are likely <b>MAR</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:22px; color:{TEXT}; text-align:center;'><b>Missingness in Nutritional Deficiencies by Age Range</b></p>", unsafe_allow_html=True)

    # Ensure Age_Range exists & compute missingness indicator
    if "Age" in hair_raw.columns and "Age_Range" not in hair_raw.columns:
        hair_raw['Age_Range'] = pd.cut(hair_raw['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
    hair_raw['Nutritional_Deficiencies_missing'] = hair_raw['Nutritional_Deficiencies'].isna().astype(int)

    age_sorted = hair_raw.sort_values('Age_Range', na_position='last')
    matrix_age = age_sorted["Nutritional_Deficiencies_missing"].to_numpy().reshape(1, -1)
    fig, ax = plt.subplots(figsize=(10,2))
    sns.heatmap(matrix_age, cmap="YlGn", cbar=True, ax=ax)
    uniq_age = list(age_sorted["Age_Range"].dropna().unique())
    xticks = [np.mean(np.where(age_sorted["Age_Range"] == lvl)) for lvl in uniq_age]
    ax.set_xticks(xticks)
    ax.set_xticklabels(uniq_age, fontsize=11, rotation=0)
    ax.set_yticks([0])
    ax.set_yticklabels(["Nutritional_Deficiencies_missing"], fontsize=11)
    ax.set_title("By Age Range", fontsize=14, color=HEADER_COLOR)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Chi-squared test
    st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Chi-Squared Tests for Missingness</h2>", unsafe_allow_html=True)
    contingency_age = pd.crosstab(hair_raw['Nutritional_Deficiencies_missing'], hair_raw['Age_Range'])
    chi2_age, p_age, dof_age, ex_age = stats.chi2_contingency(contingency_age)

    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        The missingness in <b>Nutritional_Deficiencies</b> was tested against <b>Age Range</b>.<br>
        The analysis confirms that missingness is <b>not random</b>.<br><br>
        - p-value (Age Range): <b>{p_age:.3e}</b><br><br>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Bar: observed vs expected for Age
    x = np.arange(len(contingency_age.columns))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    obs = contingency_age.loc[1] if 1 in contingency_age.index else pd.Series(0,index=contingency_age.columns)
    exp = pd.Series(ex_age[1], index=contingency_age.columns) if ex_age.shape[0] > 1 else pd.Series(0,index=contingency_age.columns)
    ax.bar(x - width/2, obs, width, label='Observed', color='#A8D5BA')
    ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
    ax.set_xticks(x)
    ax.set_xticklabels(obs.index)
    ax.set_title("Missingness: Age Range", fontsize=12, color=HEADER_COLOR)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Logistic regression static table for Nutritional_Deficiencies
    st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Logistic Regression for Missingness in Nutritional Deficiencies</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        We perform a <b>logistic regression</b> to assess how <b>Age</b> relates to the probability of missing values in <b>Nutritional_Deficiencies</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <table style="width:70%; border-collapse:collapse; font-size:16px; color:{TEXT}; margin-left:auto; margin-right:auto;">
        <tr style="background-color:#f2f2f2;">
            <th> </th><th>coef</th><th>std err</th><th>z</th><th>P>|z|</th><th>[0.025</th><th>0.975]</th>
        </tr>
        <tr><td>const</td><td>4.7461</td><td>0.711</td><td>6.679</td><td>0.000</td><td>3.353</td><td>6.139</td></tr>
        <tr><td>Weight_Loss_Encoding</td><td>-0.1755</td><td>0.269</td><td>-0.653</td><td>0.514</td><td>-0.703</td><td>0.351</td></tr>
        <tr><td>Genetic_Encoding</td><td>0.1333</td><td>0.269</td><td>0.496</td><td>0.620</td><td>-0.394</td><td>0.660</td></tr>
        <tr><td>Age</td><td>-0.2659</td><td>0.029</td><td>-9.099</td><td>0.002</td><td>-0.323</td><td>-0.209</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        We can observe that the <b>p-value for Age</b> is very low (0.002), indicating that it is statistically significant. 
        This confirms that <b>Age</b> has a meaningful relationship with the missingness of <b>Nutritional_Deficiencies</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # ---- Imputation: Mode by Age Group ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >First Method: Mode (by Age Group)</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
    We can impute missing <b>Nutritional_Deficiencies</b> using a conditional mode approach by <b>Age Range</b>.
    </p>
    """, unsafe_allow_html=True)

    # Prepare and apply
    df_nut_mode = df_raw.copy()
    bins = [18, 30, 40, 51]
    labels = ['18-30', '30-40', '40-51']
    if 'Age' in df_nut_mode.columns:
        df_nut_mode['Age_Range'] = pd.cut(df_nut_mode['Age'], bins=bins, labels=labels, right=False)
    df_nut_mode['Nutritional_Deficiencies'] = df_nut_mode.apply(lambda row: impute_nutritional_mode(row, df_nut_mode), axis=1)

    # Plot before vs after mode
    count_raw_nut = df_raw['Nutritional_Deficiencies'].value_counts().sort_index()
    count_mode_nut = df_nut_mode['Nutritional_Deficiencies'].value_counts().sort_index()
    all_nut_conditions = sorted(set(count_raw_nut.index).union(set(count_mode_nut.index)))
    count_raw_nut = count_raw_nut.reindex(all_nut_conditions, fill_value=0)
    count_mode_nut = count_mode_nut.reindex(all_nut_conditions, fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(all_nut_conditions))
    width = 0.35
    ax.bar([i - width/2 for i in x], count_raw_nut.values, width, label='Before Imputation', color='#86aca9')
    ax.bar([i + width/2 for i in x], count_mode_nut.values, width, label='After Mode Imputation (Age Group)', color="#F7B65B")
    ax.set_xticks(x)
    ax.set_xticklabels(all_nut_conditions, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Nutritional Deficiencies: Before vs After Mode Imputation (by Age Group)', fontsize=16, color='#4c8179')
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)

    # ---- Random Forest comparison (visual) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >Second Method: Random Forest (Nutritional Deficiencies)</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
    Missing <b>Nutritional_Deficiencies</b> were imputed using a <b>Random Forest (n_estimators=1000)</b> model trained on features: Stress Level, Age Range, Genetic Encoding, Hormonal Changes, Smoking, Weight Loss, Environmental Factors.
    </p>
    """, unsafe_allow_html=True)

    # Prepare cleaned vs raw plotting
    df_raw_copy = df_raw.copy()
    df_cleaned_copy = df_cleaned.copy()
    if 'Nutritional_Deficiencies' in df_raw_copy.columns:
        df_raw_copy['Nutritional_Deficiencies'] = df_raw_copy['Nutritional_Deficiencies'].dropna().astype(str).str.strip().str.title()
    if 'Nutritional_Deficiencies' in df_cleaned_copy.columns:
        df_cleaned_copy['Nutritional_Deficiencies'] = df_cleaned_copy['Nutritional_Deficiencies'].dropna().astype(str).str.strip().str.title()

    count_raw_n = df_raw_copy['Nutritional_Deficiencies'].value_counts().sort_index()
    count_cleaned_n = df_cleaned_copy['Nutritional_Deficiencies'].value_counts().sort_index()
    all_nutri = sorted(set(count_raw_n.index).union(set(count_cleaned_n.index)))
    count_raw_n = count_raw_n.reindex(all_nutri, fill_value=0)
    count_cleaned_n = count_cleaned_n.reindex(all_nutri, fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(all_nutri))
    width = 0.35
    ax.bar([i - width/2 for i in x], count_raw_n.values, width, label='Before Imputation', color="#9BDEB6")
    ax.bar([i + width/2 for i in x], count_cleaned_n.values, width, label='After Imputation (Random Forest)', color="#416c65")
    ax.set_xticks(x)
    ax.set_xticklabels(all_nutri, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Nutritional Deficiencies: Before vs After Random Forest Imputation', fontsize=16, color=HEADER_COLOR)
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)

    # Observation card with your domain note (80 missing in 18-30 etc.)
    st.markdown(f"""
    <div style='background-color:#FFF4E5; padding:14px; border-radius:12px;'>
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        <b>Observation:</b> For <b>Nutritional_Deficiencies</b>, <b>Random Forest imputation</b> is preferable to simple mode imputation. 
        In our dataset the <b>only 80 missing</b> Nutritional_Deficiencies records are concentrated in <b>Age Range = 18-30</b>. 
        The most frequent label in that group is <b>Vitamin A</b>, but filling all missing entries with this single mode would be misleading. 
        We are working with a hair health dataset — <b>Vitamin A is not primarily implicated in hair loss</b>, whereas <b>Biotin</b>, <b>Zinc</b>, and <b>Vitamin D</b> are more relevant. 
        Assigning Vitamin A by default would likely misrepresent the true distribution; Random Forest uses multiple features per individual and produces more plausible imputations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # done
