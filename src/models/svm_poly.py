from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from result import Result


def run_model(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='poly', degree=2, gamma="scale")  # poly Kernel

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    return Result(result=y_pred, y_test=y_test)
