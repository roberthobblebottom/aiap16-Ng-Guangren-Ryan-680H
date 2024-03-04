from sqlite import connection_and_retrieve_data
from preprocess import columns_and_indices_names_cleaning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from pipeline import pipeline
import pprint
import sys

"""Main python script to run the program
    """
if __name__ == "__main__":
    db_path = sys.argv[1]
    df = connection_and_retrieve_data(db_path).drop_duplicates()
    df = columns_and_indices_names_cleaning(df)
    results = pipeline(RandomForestClassifier(), df.copy())
    print("Random Forest Classifier:")
    pprint.pprint(results)
    print("\n\n\n")

    print("Gradient Boosting Classifier:")
    results = pipeline(HistGradientBoostingClassifier(), df.copy())
    pprint.pprint(results)
    print("\n\n\n")

    print("Support Vector Mechanism:")
    results = pipeline(SVC(), df.copy())
    pprint.pprint(results)
    print("\n\n\n")
