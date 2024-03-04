import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

"""Module for custom preprocessing steps loggically seggregated via their roles
"""


def columns_and_indices_names_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """columns and indices cleaning for easier data wrangling in the sklearn pipeline later

    Args:
        df (pd.DataFrame): dataframe to be processed

    Returns:
        pd.DataFrame: dataframe processed
    """
    df.columns = df.columns.str.replace(" ", "_", regex=True)

    df.columns = df.columns.str.lower()
    df.set_index("id", inplace=True)
    return df


class Cleaning(ClassifierMixin, BaseEstimator):
    """Custom sklearn pipeline object for data cleaning

    explaination of how the cleaning is dones is in eda.ipynb
    Args:
        ClassifierMixin (_type_): _description_
        BaseEstimator (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[(X.age < 0) | (X.age > 105), "age"] = "0"
        X.loc[(X.start_smoking.str.len() > 4) | (X.stop_smoking.str.len() > 4)] = "0"

        c = ["start_smoking", "stop_smoking", "current_weight", "last_weight"]
        X[c] = X[c].astype("int64")
        X["gender"] = X.gender.str.lower()
        X.loc[X.gender == "nan", "gender"] = None

        X.loc[X.dominant_hand == "RightBoth", "dominant_hand"] = "Both"
        return X


class FeatureEngineering(ClassifierMixin, BaseEstimator):
    """Custom sklearn pipeline object for feature engineering


    explaination of how the cleaning is dones is in eda.ipynb
    Args:
        ClassifierMixin (class): Mixin class for all classifiers in scikit-learn.
        BaseEstimator (class): Base class for all estimators in scikit-learn.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["weight_difference"] = X.current_weight - X.last_weight

        X["start_smoking_numerical"] = X.loc[
        X["start_smoking"] != 0, "start_smoking"
        ].astype("int64")
        X["stop_smoking_numerical"] = X.loc[
            X["stop_smoking"] != 0, "stop_smoking"
        ].astype("int64")
        X["years_of_smoking"] = (
            X.stop_smoking_numerical - X.start_smoking_numerical
        ).fillna(0)


        X.loc[X.years_of_smoking >0, "has_history_of_smoking"] = True
        X.loc[X.years_of_smoking<=0,"has_history_of_smoking"] = False
        
        X.drop(
            ["start_smoking_numerical", "stop_smoking_numerical"], axis=1, inplace=True
        )
        return X
