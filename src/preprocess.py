import pandas as pd

"""Preprocessing steps loggically seggregated via their roles
"""


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    for handling of duplicates.
    Args:
        df (pd.DataFrame): DataFrame to be processed.

    Returns:
        pd.DataFrame: processed DataFrame
    """

    df.drop_duplicates(inplace=True)
    return df


def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """For cleaning of data, columns and indexes.
    Args:
        df (pd.DataFrame): DataFrame to be processed.

    Returns:
        pd.DataFrame: processed DataFrame
    """
    df.columns = df.columns.str.replace(" ", "_", regex=True)
    df.columns = df.columns.str.lower()
    df.set_index("id", inplace=True)

    df.loc[(df.age < 0) | (df.age > 105), "age"] = None

    df["gender"] = df.gender.str.lower()
    df.loc[df.gender == "nan", "gender"] = None

    df.loc[df.dominant_hand == "RightBoth", "dominant_hand"] = "Both"
    return df


def imputation(df: pd.DataFrame) -> pd.DataFrame:
    """For imputation of missing data (None, np.nan)

    Args:
        df (pd.DataFrame): DataFrame to be processed

    Returns:
        pd.DataFrame: processed DataFrame
    """
    df.interpolate(inplace=True)
    df.ffill(inplace=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """For engineering of features before putting through model

    Args:
        df (pd.DataFrame): DataFrame to be processed

    Returns:
        pd.DataFrame: processed DataFrame
    """
    df["weight_difference"] = df.current_weight - df.last_weight

    df["start_smoking_numerical"] = df.loc[
        df["start_smoking"].str.len() == 4, "start_smoking"
    ].astype("int64")
    df["stop_smoking_numerical"] = df.loc[
        df["stop_smoking"].str.len() == 4, "stop_smoking"
    ].astype("int64")
    df["years_of_smoking"] = (
        df.stop_smoking_numerical - df.start_smoking_numerical
    ).fillna(0)
    return df
