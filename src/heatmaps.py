import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missingness_heatmap_nutritional_deficiencies(df, missing_col, group_col, figsize=(12,2), cmap='YlGn', title=None):
    """
    Plots a heatmap for missing values of a column, sorted by a grouping column.
    
    Parameters:
    - df: pd.DataFrame
    - missing_col: str, column indicating missingness (e.g., Nutritional_Deficiencies_missing)
    - group_col: str, column to sort/group by (e.g., Age_Range)
    - figsize: tuple, size of the plot
    - cmap: str, colormap
    - title: str, optional plot title
    """
    # Sort by group_col
    df_sorted = df.sort_values(group_col)
    # df[missing_col] = df[missing_col].isna().astype(int)
    
    # Reshape the missingness column into 2D for heatmap
    missing_matrix = df_sorted[missing_col].to_numpy().reshape(1, -1)
    
    plt.figure(figsize=figsize)
    sns.heatmap(missing_matrix, cmap=cmap, cbar=True, linewidths=0.5)
    
    # Set x-ticks at the mean position for each group
    unique_groups = df_sorted[group_col].unique()
    xticks = [np.mean(np.where(df_sorted[group_col]==lvl)) for lvl in unique_groups]
    plt.xticks(ticks=xticks, labels=unique_groups)
    
    plt.yticks([0], [missing_col])
    
    if title:
        plt.title(title, fontsize=14)
    
    st.pyplot()
