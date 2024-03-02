from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd

random_state = 0


def modeling(
    model: BaseEstimator,
    df: pd.DataFrame,
) -> dict:
    pipeline = Pipeline(
        [
            (
                "ordinal_encoder",
                OrdinalEncoder(),
            ),
            (
                "standard_scaler",
                StandardScaler(),
            ),
            (
                "simple_imputer",
                SimpleImputer(),
            ),
            (
                "select_k_best",
                SelectKBest(),
            ),
            ("model", model),
        ]
    )
    train_x, train_y, evaluation_x, evaluation_y = train_test_split(
        df, shuffle=True, random_state=random_state, test_size=0.1
    )
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(evaluation_x, evaluation_y)
    return {
        "model": model,
        "classification_report": classification_report(evaluation_y, predictions),
        "roc_auc_score": roc_auc_score(evaluation_y, predictions),
        "confusion_matrix": confusion_matrix(evaluation_y, predictions),
    }
