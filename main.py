
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
LOGISTIC_REGRESSION = Model(
    name="Logistic Regression",
    estimator=LogisticRegression(),
    hyperparameters={
        "solver": ["newton-cg", "lbfgs", "liblinear"]
    }
)

RANDOM_FOREST = Model(
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

DECISION_TREE = Model(
    name="Decision Tree",
    estimator=DecisionTreeClassifier(),
    hyperparameters={
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 5, 8]
    }
)


if __name__ == "__main__":
    train = DataSet()
    train.load_input("train_values.csv")
    train.convert_to_one_hot()
    assert not train.has_null_values()
    train.normalize_input()
    train.load_output("train_labels.csv")

    models = [LOGISTIC_REGRESSION, RANDOM_FOREST, DECISION_TREE]
    for model in models:
        model.fit_best_hyperparameters(train)
    model = select_best_model(models)

    test = DataSet()
    test.load_input("test_values.csv")
    test.convert_to_one_hot()
    assert not test.has_null_values()
    test.normalize_input()
