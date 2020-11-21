
class Result:
    # TODO determine important metrics
    def __init__(self, test_acc=None, train_acc=None, error=None):
        self.test_acc = test_acc
        self.train_acc = train_acc
        self.error = error
