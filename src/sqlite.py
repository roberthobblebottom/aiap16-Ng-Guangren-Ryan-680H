import sqlite3
import pandas as pd

"""sqlite3 module
"""


def connection_and_retrieve_data(path: str) -> pd.DataFrame:
    """Connect to sqlite database and retrieve data in the form of DataFrame

    Args:
        path (str): database path

    Returns:
        pd.DataFrame: retrieved DataFrame
    """
    connection = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM lung_cancer", connection)
    connection.close()
    return df
