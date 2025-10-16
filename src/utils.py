def ensure_index_reset(df):
    """
    Reset index if needed and drop old index column.
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    return df
