# src/medical_missingness.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

sns.set_style("whitegrid")

def impute_medical_condition_mode(row, df):
    """
    Mode imputation grouped by Age_Range and Stress_Level.
    Assumes df already has 'Age_Range' and 'Stress_Level' columns.
    """
    if pd.isna(row['Medical_Conditions']):
        group = df[(df['Age_Range'] == row['Age_Range']) & (df['Stress_Level'] == row['Stress_Level'])]
        mode_value = group['Medical_Conditions'].mode()
        if not mode_value.empty:
            return mode_value[0]
        else:
            return 'Unknown'
    else:
        return row['Medical_Conditions']

def render_medical_missingness(hair_raw, df_raw, df_cleaned,
                               HEADER_COLOR="#2E8B57", TEXT="#333",
                               SECTION_BG="#F0F8F5", ACCENT="#2E8B57"):
    """
    Renders the full Medical Conditions missingness section in Streamlit.
    Expects:
      - hair_raw : DataFrame used for missingness analysis (has 'Stress' / 'Stress_Level', 'Age' columns).
      - df_raw : original raw file DataFrame (for before-imputation counts).
      - df_cleaned : cleaned/imputed DataFrame (for after-imputation counts).
    """

    # -- Header
    st.markdown(f"""
        <div style='background-color:{SECTION_BG}; padding:12px; text-align:center; border-radius:10px; margin-top:20px;'>
            <h3 style='color:{ACCENT}; font-size:25px; margin:6px 0;'>Missingness in Medical Conditions</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Observation card
    st.markdown(f"""
    <div style='background-color:#E6F4EA; padding:14px; border-radius:12px;'>
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        <b>Observation:</b> Missingness in <b>Medical_Conditions</b> is not completely random (not MCAR). Chi-squared tests reveal a strong dependency on <b>Stress Level</b> and <b>Age Range</b>. Higher stress and older age correlate with more missing entries. Thus, these values are likely <b>MAR</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Ensure encoding & Age_Range exist
    if "Stress" in hair_raw.columns and "Stress_Level" not in hair_raw.columns:
        hair_raw['Stress_Level'] = hair_raw['Stress']
    if "Age" in hair_raw.columns and "Age_Range" not in hair_raw.columns:
        hair_raw['Age_Range'] = pd.cut(hair_raw['Age'], bins=[18,30,40,51], labels=['18-30','30-40','40-51'], right=False)
    hair_raw["Medical_Conditions_missing"] = hair_raw["Medical_Conditions"].isna().astype(int)

    # Missingness plots: Stress & Age
    st.markdown(f"<p style='font-size:22px; color:{TEXT}; text-align:center;'><b>Missingness in Medical Conditions by Stress Level & Age Range</b></p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    # Heatmap by Stress
    with col1:
        stress_order = ["Low", "Moderate", "High"]
        hair_raw["Stress_Level"] = pd.Categorical(hair_raw["Stress_Level"], categories=stress_order, ordered=True)
        hair_sorted = hair_raw.sort_values("Stress_Level")
        matrix_stress = hair_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)
        fig, ax = plt.subplots(figsize=(10,2))
        sns.heatmap(matrix_stress, cmap="YlGn", cbar=True, ax=ax)
        xticks = [np.mean(np.where(hair_sorted["Stress_Level"] == lvl)) for lvl in stress_order if lvl in hair_sorted["Stress_Level"].unique()]
        ax.set_xticks(xticks)
        ax.set_xticklabels([lvl for lvl in stress_order if lvl in hair_sorted["Stress_Level"].unique()], fontsize=11, rotation=0)
        ax.set_yticks([0])
        ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
        ax.set_title("By Stress Level", fontsize=14, color=HEADER_COLOR)
        st.pyplot(fig)
        plt.close(fig)

    # Heatmap by Age
    with col2:
        age_sorted = hair_raw.sort_values("Age_Range", na_position='last')
        matrix_age = age_sorted["Medical_Conditions_missing"].to_numpy().reshape(1, -1)
        fig, ax = plt.subplots(figsize=(10,2))
        sns.heatmap(matrix_age, cmap="YlGn", cbar=True, ax=ax)
        uniq_age = list(age_sorted["Age_Range"].dropna().unique())
        xticks = [np.mean(np.where(age_sorted["Age_Range"] == lvl)) for lvl in uniq_age]
        ax.set_xticks(xticks)
        ax.set_xticklabels(uniq_age, fontsize=11, rotation=0)
        ax.set_yticks([0])
        ax.set_yticklabels(["Medical_Conditions Missing"], fontsize=11)
        ax.set_title("By Age Range", fontsize=14, color=HEADER_COLOR)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Chi-squared summary (static text + actual calculation)
    st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Chi-Squared Tests for Missingness</h2>", unsafe_allow_html=True)
    contingency_stress = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Stress_Level'])
    contingency_age = pd.crosstab(hair_raw['Medical_Conditions_missing'], hair_raw['Age_Range'])
    chi2_stress, p_stress, dof_stress, ex_stress = stats.chi2_contingency(contingency_stress)
    chi2_age, p_age, dof_age, ex_age = stats.chi2_contingency(contingency_age)

    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        The missingness in <b>Medical Conditions</b> was tested against both <b>Stress Level</b> and <b>Age Range</b>.<br>
        The analysis confirms that missingness is <b>not random</b>.<br><br>
        - p-value (Stress Level): <b>{p_stress:.3e}</b><br>
        - p-value (Age Range): <b>{p_age:.3e}</b><br><br>
        These extremely low probabilities indicate a strong relationship between missingness and these variables.
        </p>
        """,
        unsafe_allow_html=True
    )

    x = np.arange(len(contingency_stress.columns))
    width = 0.35
    col1, col2 = st.columns(2, gap="large")

    # Stress Level bar: observed vs expected
    with col1:
        fig, ax = plt.subplots(figsize=(8,5))
        obs = contingency_stress.loc[1] if 1 in contingency_stress.index else pd.Series(0,index=contingency_stress.columns)
        exp = pd.Series(ex_stress[1], index=contingency_stress.columns) if ex_stress.shape[0] > 1 else pd.Series(0,index=contingency_stress.columns)
        ax.bar(x - width/2, obs, width, label='Observed', color='#86aca9')
        ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
        ax.set_xticks(x)
        ax.set_xticklabels(obs.index)
        ax.set_title("Missingness: Stress Level", fontsize=14, color=HEADER_COLOR)
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Age Range bar
    with col2:
        fig, ax = plt.subplots(figsize=(8,5))
        obs = contingency_age.loc[1] if 1 in contingency_age.index else pd.Series(0,index=contingency_age.columns)
        exp = pd.Series(ex_age[1], index=contingency_age.columns) if ex_age.shape[0] > 1 else pd.Series(0,index=contingency_age.columns)
        ax.bar(x - width/2, obs, width, label='Observed', color='#A8D5BA')
        ax.bar(x + width/2, exp, width, label='Expected', color='#5d9189')
        ax.set_xticks(x)
        ax.set_xticklabels(obs.index)
        ax.set_title("Missingness: Age Range", fontsize=14, color=HEADER_COLOR)
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # Logistic regression static table for Medical Conditions (keeps same style)
    st.markdown(f"<h2 style='color:{HEADER_COLOR}; font-size:22px; text-align:center'; >Logistic Regression for Missingness in Medical Conditions</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        Moreover, we perform a <b>logistic regression</b> to evaluate the relationship between <b>Stress Level</b> and the probability 
        of missing values in <b>Medical Conditions</b>.
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
            <tr><td>const</td><td>-2.7795</td><td>0.247</td><td>-11.264</td><td>0.000</td><td>-3.263</td><td>-2.296</td></tr>
            <tr><td>Weight_Loss_Encoding</td><td>0.0817</td><td>0.204</td><td>0.401</td><td>0.688</td><td>-0.317</td><td>0.481</td></tr>
            <tr><td>Genetic_Encoding</td><td>0.3407</td><td>0.207</td><td>1.649</td><td>0.099</td><td>-0.064</td><td>0.746</td></tr>
            <tr><td>Stress_Level</td><td>0.4212</td><td>0.130</td><td>3.243</td><td>0.001</td><td>0.167</td><td>0.676</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <p style="font-size:18px; color:{TEXT}; line-height:1.5;">
        We can observe that the <b>p-value for Stress_Level</b> is very low (0.001), indicating that it is statistically significant. 
        This confirms that <b>Stress Level</b> has a meaningful relationship with the missingness of <b>Medical Conditions</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)

    # ---- Imputation: Mode (conditional) ----
    st.markdown(
        f"""
        <div style='background-color:#C7E9C0; padding:12px; border-radius:12px; text-align:center;'>
            <h2 style='font-size:25px; color:#2E8B57; margin:0;'>Imputation of Missing Values</h2>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >First Method: Mode (Conditional)</h2>", unsafe_allow_html=True)

    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
    Missing <b>Medical_Conditions</b> can be imputed using a <b>conditional mode</b> approach. 
    Here, each missing value is filled with the most frequent condition observed within its subgroup defined by <b>Age Range</b> 
    and <b>Stress Level</b>.
    </p>
    """, unsafe_allow_html=True)

    # Apply mode-based imputation safely: prepare columns first
    df_mode = df_raw.copy()
    bins = [18, 30, 40, 51]
    labels = ['18-30', '30-40', '40-51']
    if 'Age' in df_mode.columns:
        df_mode['Age_Range'] = pd.cut(df_mode['Age'], bins=bins, labels=labels, right=False)
    if 'Stress' in df_mode.columns:
        df_mode['Stress_Level'] = df_mode['Stress']
    # apply imputation
    df_mode['Medical_Conditions'] = df_mode.apply(lambda row: impute_medical_condition_mode(row, df_mode), axis=1)

    # Plot before vs after (mode)
    count_mode = df_mode['Medical_Conditions'].value_counts().sort_index()
    count_raw = df_raw['Medical_Conditions'].value_counts().sort_index()
    all_conditions = sorted(set(count_raw.index).union(set(count_mode.index)))
    count_raw = count_raw.reindex(all_conditions, fill_value=0)
    count_mode = count_mode.reindex(all_conditions, fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(all_conditions))
    width = 0.35
    ax.bar([i - width/2 for i in x], count_raw.values, width, label='Before Imputation', color='#86aca9')
    ax.bar([i + width/2 for i in x], count_mode.values, width, label='After Mode Imputation', color='#F7B65B')
    ax.set_xticks(x)
    ax.set_xticklabels(all_conditions, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Medical Conditions: Before vs After Mode Imputation', fontsize=16, color=HEADER_COLOR)
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)

    # ---- Random Forest Imputation (visual comparison) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #AAA; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{TEXT}; font-size:22px; text-align:center'; >Second Method: Random Forest</h2>", unsafe_allow_html=True)

    st.markdown(f"""
    <p style='font-size:18px; color:{TEXT}; line-height:1.5;'>
    Missing <b>Medical_Conditions</b> were imputed using <b>Random Forest (n_estimators=1000)</b> 
    trained on features: Stress Level, Age Range, Genetic Encoding, Hormonal Changes, Smoking, Weight Loss, Environmental Factors.
    </p>
    """, unsafe_allow_html=True)

    # prepare and plot raw vs cleaned
    df_raw_copy = df_raw.copy()
    df_cleaned_copy = df_cleaned.copy()
    if 'Medical_Conditions' in df_raw_copy.columns:
        df_raw_copy['Medical_Conditions'] = df_raw_copy['Medical_Conditions'].dropna().astype(str).str.strip().str.title()
    if 'Medical_Conditions' in df_cleaned_copy.columns:
        df_cleaned_copy['Medical_Conditions'] = df_cleaned_copy['Medical_Conditions'].dropna().astype(str).str.strip().str.title()

    count_raw = df_raw_copy['Medical_Conditions'].value_counts().sort_index()
    count_cleaned = df_cleaned_copy['Medical_Conditions'].value_counts().sort_index()
    all_conditions = sorted(set(count_raw.index).union(set(count_cleaned.index)))
    count_raw = count_raw.reindex(all_conditions, fill_value=0)
    count_cleaned = count_cleaned.reindex(all_conditions, fill_value=0)

    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(all_conditions))
    width = 0.35
    ax.bar([i - width/2 for i in x], count_raw.values, width, label='Before Imputation', color="#9BDEB6")
    ax.bar([i + width/2 for i in x], count_cleaned.values, width, label='After Imputation', color="#416c65")
    ax.set_xticks(x)
    ax.set_xticklabels(all_conditions, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Medical Conditions: Before vs After Imputation (Random Forest)', fontsize=16, color=HEADER_COLOR)
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)

    # Observation card
    st.markdown(f"""
    <div style='background-color:#FFF4E5; padding:14px; border-radius:12px;'>
        <p style='font-size:18px; color:{TEXT}; margin:0;'>
        <b>Observation:</b> In this scenario, <b>Random Forest imputation</b> is more suitable than <b>Mode imputation</b>. 
        For example, in the group with <b>Stress Level = 2</b> and <b>Age Range = 18-30</b>, there is no single dominant <b>Medical_Condition</b>; counts are very close. 
        Mode imputation would arbitrarily assign the most frequent value, potentially introducing bias. 
        Random Forest can leverage multiple features simultaneously to make more informed predictions, preserving the underlying distribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # done
