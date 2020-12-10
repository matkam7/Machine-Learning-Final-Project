from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from result import Result

from sklearn.preprocessing import MinMaxScaler


def run_model(x_train, x_test, y_train, y_test):
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    logreg = LogisticRegression(solver="lbfgs").fit(x_train, y_train)
    predited = logreg.predict(x_test)
    return Result(result=predited, y_test=y_test)
