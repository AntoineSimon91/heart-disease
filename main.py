
# third-party imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# local imports
from learning.dataset import DataSet
from learning.models import Model, select_best_model

# setup pandas dataframes display options
pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 120)

# initialize machine learning models
logistic_regression = Model(
    name="Logistic Regression",
    estimator=LogisticRegression(),
    hyperparameters={"solver": ["newton-cg", "lbfgs", "liblinear"]}
)

random_forest = Model(
    name="Random Forest",
    estimator=RandomForestClassifier(),
    hyperparameters={
        "n_estimators": [4, 6, 9],
        "criterion": ["entropy", "gini"],
        "max_depth": [2, 5, 10],
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 5, 8],
        "min_samples_split": [2, 3, 5]
    }
)

decision_tree = Model(
    name="Decision Tree",
    estimator=DecisionTreeClassifier(),
    hyperparameters={
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 5, 8]
    }
)

# machine learning pipeline
train = DataSet()
train.load_input("train_values.csv")
train.convert_to_one_hot(
    converter={
        'slope_of_peak_exercise_st_segment': 'slope',
        'thal': None,
        'chest_pain_type': 'chest_pain',
        'resting_ekg_results': 'resting_ekg',
        'num_major_vessels': 'num_major_vessels'
    }
)
assert not train.has_null_values()
train.normalize_input()
train.load_output("train_labels.csv")
models = [logistic_regression, decision_tree, random_forest]
model = select_best_model(train, models)
