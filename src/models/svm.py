from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

from result import Result

def run_model(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, y_pred)
    return Result(test_acc=test_acc)

