from sklearn.metrics import accuracy_score
import sklearn as sk


class Result:
    # TODO determine important metrics
    def __init__(self, result=None, y_test=None, test_acc_mean=None, test_acc_std=None, error=None, outputs=None):

        if error != None:
            self.error = error
            return
        elif result is not None:
            self.result = result

            accuracy = accuracy_score(y_test, result)
            # Precision
            precision = sk.metrics.precision_score(
                y_test, result,  average='binary')
            # Recall
            recall = sk.metrics.recall_score(
                y_test, result, average='binary')

            # ROC AUC
            roc_auc = sk.metrics.roc_auc_score(y_test, result)
            # Confusion Matrix
            confusion_mtx = sk.metrics.confusion_matrix(
                y_test, result)

            f_score = sk.metrics.f1_score(
                y_test, result, average='micro')

            self.output = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "AUC": roc_auc,
                "F-score": f_score,
                "Confusion Matrix": str(confusion_mtx).replace("\n", ",")
            }
            self.test_acc = accuracy
            self.error = None
        else:
            self.test_acc_mean = test_acc_mean
            self.test_acc_std = test_acc_std
            self.outputs = outputs
            self.error = error
