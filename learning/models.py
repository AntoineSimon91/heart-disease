from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

MODELS = [
    {
        "name": "logistic regression",
        "estimator": LogisticRegression(),
        "hyperparameters": {
            "solver": ["newton-cg", "lbfgs", "liblinear"]
        }
    },
    {
        "name": "decision tree",
        "estimator": DecisionTreeClassifier(),
        "hyperparameters": {
            "max_features": ["log2", "sqrt"],
            "min_samples_leaf": [1, 5, 8]
        }
    },
    {
        "name": "random forest",
        "estimator": RandomForestClassifier(),
        "hyperparameters":
        {
            "n_estimators": [4, 6, 9],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 5, 10],
            "max_features": ["log2", "sqrt"],
            "min_samples_leaf": [1, 5, 8],
            "min_samples_split": [2, 3, 5]
        }
    }
]


def select_model(training_set):
    assert hasattr(training_set, "input")
    assert hasattr(training_set, "output")

    for model in MODELS:
        print(model["name"])

        grid_search = GridSearchCV(
            estimator=model["estimator"],
            param_grid=model["hyperparameters"],
            cv=KFold(10, shuffle=True),
            scoring="neg_log_loss"
        )

        grid_search.fit(training_set.input, training_set.output)

        model["best_model_param"] = grid_search.best_params_
        model["best_model_score"] = -grid_search.best_score_
        model["best_model_estimator"] = grid_search.best_estimator_

        print("Best Score: {}".format(model["best_model_score"]))
        print("Best Parameters: {}\n".format(model["best_model_param"]))
