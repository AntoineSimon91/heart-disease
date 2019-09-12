
# third party imports
from sklearn.model_selection import KFold, GridSearchCV


def select_best_model(models):
    print("Select best model (estimator, hyperparameters):")
    best_model = None
    global_best_score = None

    for model in models:
        if global_best_score is None:
            global_best_score = model.best_score

        elif model.best_score < global_best_score:
            global_best_score = model.best_score
            best_model = model

    return best_model


class Model:
    def __init__(self, name, estimator, hyperparameters):
        self.name = name
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.cross_validation = KFold(10, shuffle=True)
        self.scoring_function = "neg_log_loss"

    def fit_best_hyperparameters(self, training_set):
        assert hasattr(training_set, "input")
        assert hasattr(training_set, "output")

        print(f"  {self.name}:")

        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.hyperparameters,
            cv=self.cross_validation,
            scoring=self.scoring_function
        )

        grid_search.fit(training_set.input, training_set.output)

        self.best_hyperparameters = grid_search.best_params_
        self.best_score = -grid_search.best_score_
        self.estimator = grid_search.best_estimator_

        print(f"    Best Score: {self.best_score:.2f}")
        print(f"    Best Parameters: {self.best_hyperparameters}")

        return self.best_score
