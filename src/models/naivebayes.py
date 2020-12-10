import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from result import Result

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def run_model(x_train, x_test, y_train, y_test):
    gnb = GaussianNB(priors=None,var_smoothing= 1e-100)
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    return Result(result=y_pred, y_test=y_test)
