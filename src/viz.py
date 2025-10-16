import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def missing_values_heatmap_matrix(df: pd.DataFrame, subset_cols: list) -> plt.Figure:
    """
    Create heatmap showing missingness pattern (transposed) similar to notebook.
    Returns matplotlib Figure object.
    """
    nan_array = df[subset_cols].isna().astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_xlabel('Index')
    ax.set_yticks(range(len(subset_cols)))
    ax.set_yticklabels(subset_cols)
    ax.set_title('Missing Values Heatmap')
    return fig

def observed_vs_expected_bar(observed: pd.Series, expected: pd.Series, labels: list, title="Observed vs Expected"):
    """
    Create bar chart comparing observed vs expected counts.
    observed, expected are pandas Series aligned on same index.
    Returns Figure.
    """
    x = np.arange(len(observed))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width/2, observed, width, label='Observed')
    ax.bar(x + width/2, expected, width, label='Expected')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    return fig

def pivot_heatmap(df: pd.DataFrame, index='Age_Range', columns='Stress_Level', value='Medical_Conditions_missing'):
    """
    Create annotated heatmap from pivot table of missingness counts.
    """
    pivot_table = df.pivot_table(index=index, columns=columns, values=value, aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis', ax=ax)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_title(f'Missingness of {value} by {index} and {columns}')
    return fig
