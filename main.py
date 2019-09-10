
# standard imports
from pathlib import Path

# third-party imports
import pandas as pd
import numpy as np

# local imports
from visualization.display import display_correlation_matrix
from models import select_model


pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 120)

ONE_HOT_CONVERSION = {
    'slope_of_peak_exercise_st_segment': 'slope',
    'thal': None,
    'chest_pain_type': 'chest_pain',
    'resting_ekg_results': 'resting_ekg',
    'num_major_vessels': 'num_major_vessels'
}


def load_dataframe(filename):
    dirpath = Path("data")
    filepath = dirpath / filename
    df = pd.read_csv(filepath, index_col=0)
    return df


def convert_to_one_hot(df):
    """Convert train and test dataframe columns
    to one-hot vector.

    Args:
        df (pd.DataFrame): Dataframe.

    Return:
        df (pd.DataFrame): Dataframe
    """
    for column, prefix in ONE_HOT_CONVERSION.items():
        one_hot_df = pd.get_dummies(df[column], prefix=prefix)
        df = df.join([one_hot_df])
        df = df.drop([column], axis="columns")
    return df


def no_null_values(df):
    """Avoid having null values during learing process.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        no_null (bool): True if there's no null
            values in the dataframe
    """
    no_null = not df.isnull().values.any()
    return no_null


def find_numerical_columns(df):
    serie = df.isin([0, 1]).all(axis="index")
    columns = serie[serie == False].index.values
    return columns


def normalize_numerical_columns(df):
    non_boolean_columns = find_numerical_columns(df)
    for column in non_boolean_columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


df = load_dataframe(filename="train_values.csv")
df = convert_to_one_hot(df)
assert no_null_values(df)
df = normalize_numerical_columns(df)

df=df.drop(["slope_1","normal","resting_ekg_0","chest_pain_3","num_major_vessels_0"],axis=1)

col_init=df.columns

df_labels_train=load_dataframe(filename="train_labels.csv")

df=pd.merge(df,df_labels_train,on="patient_id",how="inner")

print(select_model(df,col_init))
