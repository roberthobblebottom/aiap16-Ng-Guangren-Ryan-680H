from sqlite import connection_and_retrieve_data
from preprocess import columns_and_indices_cleaning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from pipeline import pipeline
import pprint

if __name__ == "__main__":
    df = connection_and_retrieve_data("../data/lung_cancer.db").drop_duplicates()
    df = columns_and_indices_cleaning(df)
    results = pipeline(RandomForestClassifier(), df)
    pprint.pprint(results)
