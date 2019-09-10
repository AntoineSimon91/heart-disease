from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


kf = KFold(10, shuffle=True)

def select_model(df,cols):

	dictionnaires=[
		{
		"name" : "logistic regression",
		"model" : LogisticRegression(),
		"hyperparameters" :
			{
				"solver" : ["newton-cg", "lbfgs", "liblinear"]
			}
		},
		{
		"name" : "decision tree",
		"model" : DecisionTreeClassifier(),
		"hyperparameters" :
			{
				"max_features" : ["log2","sqrt"],
	            "min_samples_leaf" : [1,5,8]
			}

		},
		{
		"name" : "random forest",
		"model" : RandomForestClassifier(),
		"hyperparameters" :
			{
				 "n_estimators" : [4,6,9],
	             "criterion" : ["entropy", "gini"],
	             "max_depth" : [2,5,10],
	             "max_features" : ["log2","sqrt"],
	             "min_samples_leaf" : [1,5,8],
	             "min_samples_split" : [2,3,5]
			}

		}
	]

	for item in dictionnaires:
		print(item["name"])
		obj = GridSearchCV(item["model"],item["hyperparameters"],cv=kf, scoring="neg_log_loss")
		obj.fit(df[cols],df["heart_disease_present"])
		item["best_model_param"]=obj.best_params_
		item["best_model_score"]=-obj.best_score_
		item["best_model_estimator"]=obj.best_estimator_
		print("Best Score: {}".format(item["best_model_score"]))
		print("Best Parameters: {}\n".format(item["best_model_param"]))