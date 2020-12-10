from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from result import Result
from sklearn.metrics import accuracy_score


def run_model(x_train, x_test, y_train, y_test):

    neigh = KNeighborsClassifier(n_neighbors=11)

    neigh.fit(x_train, y_train)

    # A = neigh.kneighbors_graph(x_train)
    # print(A.toarray())

    y_pred = neigh.predict(x_test)

    return Result(result=y_pred, y_test=y_test)
