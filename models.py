from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

kf=KFold(10,shuffle=True)

def logreg(df,col_init):
	lr=LogisticRegression(solver="liblinear")
	log_loss_logistic=-np.mean(cross_val_score(lr,df[col_init],df["heart_disease_present"],scoring="neg_log_loss",cv=kf))
	return log_loss_logistic

def decisiontree(df,col_init,min_samples_leaf,max_features):
	clf=DecisionTreeClassifier(
		min_samples_leaf=min_samples_leaf, 
		splitter="random", 
		max_features=max_features
		)
	log_loss_decision_tree=-np.mean(cross_val_score(clf,df[col_init],df["heart_disease_present"],scoring="neg_log_loss",cv=kf))
	return log_loss_decision_tree

def randomforest(df,col_init,n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf):
	clf2=RandomForestClassifier(
		n_estimators=n_estimators, 
		max_depth=max_depth, 
		min_samples_split=min_samples_split, 
		min_samples_leaf=min_samples_leaf
		)
	log_loss_random_forest=-np.mean(cross_val_score(clf2,df[col_init],df["heart_disease_present"],scoring="neg_log_loss",cv=kf))
	return log_loss_random_forest


