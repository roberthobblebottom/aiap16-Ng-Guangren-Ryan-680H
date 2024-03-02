from sqlite_wrapper import connection_and_retrieve_data
from preprocess import handle_duplicates, cleaning, imputation, feature_engineering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

if __name__ == "__main__":
    df = connection_and_retrieve_data("data/lung_cancer.db")
    df = handle_duplicates(df)
    df = cleaning(df)
    df = imputation(df)
    df = feature_engineering(df)
    model
