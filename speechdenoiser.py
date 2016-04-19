from evaluator import Evaluator
from loader import Loader


class SpeechDenoiser:
    def __init__(self, model, datafile):
        self.model = model
        self.loader = Loader()

    def prep(self):
        pass

    def train(self):
        self.model.train()
        self.model.save()

    def show(self):
        self.evaluator = Evaluator(self.model)
        self.evaluator.evaluate(self.loader)
        self.model.show()

    def close(self):
        self.evaluator.save()
        self.model.save()
