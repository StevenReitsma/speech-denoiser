class Params():
	def __init__(self):
		self.BATCH_SIZE = 2
		self.START_LEARNING_RATE = 0.001
		self.EPOCHS = 100

		self.MAX_LENGTH = 10000

		self.N_PRODUCERS = 4
		self.MULTIPROCESS = False

params = Params()
