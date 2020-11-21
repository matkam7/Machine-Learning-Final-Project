from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from result import Result


def run_model(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression().fit(x_train, y_train)
    predited = logreg.predict(x_test)
    test_acc = accuracy_score(y_test, predited)
    return Result(test_acc=test_acc)
