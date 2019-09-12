
# third-party imports
import pandas as pd

# local imports
from learning.dataset import DataSet
from learning.models import select_model


pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 120)

one_hot_converter = {
    'slope_of_peak_exercise_st_segment': 'slope',
    'thal': None,
    'chest_pain_type': 'chest_pain',
    'resting_ekg_results': 'resting_ekg',
    'num_major_vessels': 'num_major_vessels'
}

train = DataSet()
train.load_input("train_values.csv")
train.convert_to_one_hot(one_hot_converter)
assert train.no_null_values()
train.normalize_input()
train.load_output("train_labels.csv")
select_model(train)
