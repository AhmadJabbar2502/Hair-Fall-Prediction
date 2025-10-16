import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm

def missingness_indicator(df: pd.DataFrame, col='Medical_Conditions') -> pd.DataFrame:
    """Add Medical_Conditions_missing column (0/1) and return df copy."""
    df = df.copy()
    df[f"{col}_missing"] = df[col].isna().astype(int)
    return df

def chi2_test_between(df: pd.DataFrame, missing_col: str, factor_col: str) -> tuple:
    """
    Run chi2 on contingency table between missing indicator and a factor column.
    Returns (chi2, p_value, dof, expected)
    """
    contingency = pd.crosstab(df[missing_col], df[factor_col])
    chi2, p, dof, ex = chi2_contingency(contingency)
    return chi2, p, dof, ex

def logistic_missingness(df: pd.DataFrame, y_col='Medical_Conditions_missing', x_cols=None):
    """
    Fit logistic regression predicting missingness (Logit). Return fitted model.
    x_cols: list of predictor column names. Adds constant automatically.
    """
    if x_cols is None:
        x_cols = ['Weight_Loss_Encoding', 'Genetic_Encoding', 'Stress_Level']
    X = df[x_cols]
    y = df[y_col]
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)  # disp=False silences output
    return model
