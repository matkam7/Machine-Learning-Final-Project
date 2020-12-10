import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from result import Result


def run_model(x_train, x_test, y_train, y_test):

    model = AdaBoostClassifier(n_estimators=100)
    predicted = model.fit(x_train, y_train).predict(x_test)

    return Result(result=predicted, y_test=y_test)
