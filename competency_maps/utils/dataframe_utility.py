import pandas as pd


def explode(df, col_to_explode, new_col_name, fill_value=""):
    """Explode a Dataframe using the specified column."""
    columns = df.columns.drop(col_to_explode)
    col_series = df[col_to_explode].apply(pd.Series)
    df = col_series.merge(df, right_index=True, left_index=True)
    df = df.drop([col_to_explode], axis=1)
    df = (
        df.melt(id_vars=columns, value_name=new_col_name)
        .drop("variable", axis=1)
        .dropna(how="all")
    )
    # df.reset_index(inplace=True)
    return df
