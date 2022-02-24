import pandas as pd


def print_dataframe(df: pd.DataFrame, max_cols: int = 10, hide_index: bool = True):
    """Prints passed pandas dataframe in a controlled way with some sensible formatting,
    maximum number of columns, and, if needed, without the index.

    This is needed for easier doc-testing of the code snippets in the documentation.
    """
    if hide_index:
        df = df.copy()
        df.index = [""] * len(df)
    with pd.option_context(
        "display.float_format", "{:.1e}".format,
        "display.max_rows", 10,
        "display.max_columns", max_cols,
        "display.expand_frame_repr", False
    ):
        print(df)
