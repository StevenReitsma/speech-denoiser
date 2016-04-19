import cPickle as pickle


class Evaluator:
    def __init__(self, model, error):
        self.errors = []
        self.model = model
        self.error = error

    def evaluate(self, data_loader):
        data_loader.test_mode()

        for batch in data_loader.batches:
            predictions = self.model.test(batch.data())
            self.errors.append(self.error(predictions, batch.truth))

    def save(self):
        pickle.dump(self.model, open("models/model"))
        pickle.dump(self.errors, open("results/result"))

    def load(self, number):
        self.model = pickle.load("models/model{}".format(number))
        self.errors = pickle.load("results/result{}".format(number))

    def show(self):
        pass
