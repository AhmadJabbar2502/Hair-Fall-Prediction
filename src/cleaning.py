import pandas as pd
import numpy as np

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to standardized names and replace spaces with underscores.
    Returns modified copy.
    """
    columns = ['Id', 'Genetics', 'Hormonal_Changes', 'Medical_Conditions',
               'Medications_and_Treatments', 'Nutritional_Deficiencies', 'Stress',
               'Age', 'Poor_Hair_Care_Habits', 'Environmental_Factors', 'Smoking',
               'Weight_Loss', 'Hair_Loss']
    df = df.copy()
    df.columns = columns  # be cautious: ensure same number of columns
    df.columns = [col.replace(' ', '_') for col in df.columns]
    return df

def replace_no_data_with_na(df: pd.DataFrame, values=('No Data', 'No data')) -> pd.DataFrame:
    """Replace strings representing missing values with pd.NA and return copy."""
    df = df.copy()
    df.replace(list(values), pd.NA, inplace=True)
    return df

def encode_binary_columns(df: pd.DataFrame, mapping=None) -> pd.DataFrame:
    """
    Create encoded columns for binary Yes/No columns:
    - Genetics -> Genetic_Encoding
    - Hormonal_Changes -> Hormonal_Encoding ...
    Returns df copy with new columns.
    """
    df = df.copy()
    if mapping is None:
        mapping = {'Yes': 1, 'No': 0}
    df['Genetic_Encoding'] = df['Genetics'].map(mapping)
    df['Hormonal_Encoding'] = df['Hormonal_Changes'].map(mapping)
    df['Poor_Hair_Care_Encoding'] = df['Poor_Hair_Care_Habits'].map(mapping)
    df['Environmental_Encoding'] = df['Environmental_Factors'].map(mapping)
    df['Smoking_Encoding'] = df['Smoking'].map(mapping)
    df['Weight_Loss_Encoding'] = df['Weight_Loss'].map(mapping)
    return df

def clean_trailing_spaces(df: pd.DataFrame, cols=('Medical_Conditions','Medications_and_Treatments','Nutritional_Deficiencies')) -> pd.DataFrame:
    """Strip whitespace for listed columns."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("string").str.strip()
    return df

def map_stress_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'Low','Moderate','High' to 0,1,2 as Stress_Level. Returns copy."""
    df = df.copy()
    df['Stress_Level'] = df['Stress'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    return df

def create_age_ranges(df: pd.DataFrame, bins=(18,30,40,51), labels=('18-30','30-40','40-51')) -> pd.DataFrame:
    """Create Age_Range column and return copy."""
    df = df.copy()
    df['Age_Range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df
