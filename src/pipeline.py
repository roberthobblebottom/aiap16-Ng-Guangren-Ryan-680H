from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
from preprocess import FeatureEngineering, Cleaning
from sklearn.compose import ColumnTransformer

random_state = 0
"""Pipeline module
    """


def pipeline(
    model: BaseEstimator,
    df: pd.DataFrame,
) -> dict:
    features_names = df.drop("lung_cancer_occurrence", axis=1).columns
    numerical_features = [
        "age",
        "last_weight",
        "current_weight",
        "start_smoking",
        "stop_smoking",
    ]
    """pipeline

    Returns:
        dict: Includes model, predictions, various performance metrics
    """
    categorial_features = list(set(features_names) - set(numerical_features))
    pipeline = Pipeline(
        [
            ("cleaning", Cleaning()),
            ("feature_engineering", FeatureEngineering()),
            (
                "column_encoders",
                ColumnTransformer(
                    [
                        ("categorial_encoder", OrdinalEncoder(), categorial_features),
                        ("passthrough", "passthrough", numerical_features),
                    ]
                ),
            ),
            ("standard_scaler", StandardScaler()),
            ("simple_imputer", SimpleImputer(strategy="most_frequent")),
            ("select_k_best", SelectKBest(k=10)),  # choosing 10 out of the 13 features
            ("model", model),
        ]
    )
    train, evaluation = train_test_split(
        df, shuffle=True, random_state=random_state, test_size=0.1
    )
    train_x = train.drop("lung_cancer_occurrence", axis=1)
    train_y = train.lung_cancer_occurrence
    evaluation_x = evaluation.drop("lung_cancer_occurrence", axis=1)
    evaluation_y = evaluation.lung_cancer_occurrence
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(evaluation_x)
    return {
        "model": model,
        "predictions": predictions,
        "classification_report": classification_report(evaluation_y, predictions),
        "roc_auc_score": roc_auc_score(evaluation_y, predictions),
        "confusion_matrix": confusion_matrix(evaluation_y, predictions),
    }
