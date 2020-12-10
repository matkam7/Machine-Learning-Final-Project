from sklearn import tree
import numpy as np
from result import Result
from sklearn.metrics import accuracy_score


def run_model(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob_pred = clf.predict_proba(x_test.round())

    return Result(result=y_pred, y_test=y_test)
