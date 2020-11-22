
class Result:
    # TODO determine important metrics
    def __init__(self, test_acc=None, train_acc=None, error=None, test_acc_mean=None, test_acc_std=None):
        self.test_acc = test_acc
        self.train_acc = train_acc
        self.error = error
        self.test_acc_mean = test_acc_mean
        self.test_acc_std = test_acc_std

