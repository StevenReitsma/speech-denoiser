import numpy as np
from params import *
from multiprocessing import Process, Queue, JoinableQueue, Value
from threading import Thread
import theano
from time import time
import socket


class ParallelBatchIterator(object):
	"""
	Uses a producer-consumer model to prepare batches on the CPU while training on the GPU.
	"""

	def __init__(self, X, y, batch_size, dataset):
		self.batch_size = batch_size
		self.X = X
		self.y = y
		self.dataset = dataset

	def chunks(self, l, n):
		""" Yield successive n-sized chunks from l.
			from http://goo.gl/DZNhk
		"""
		for i in xrange(0, len(l), n):
			yield l[i:i + n]

	def read_data(self, filename):
		with open(filename, 'rb') as f:
			data = np.fromfile(f, dtype='>i2')

		return data / (0.001 + np.max(np.abs(data)))

	def process(self, key_x, key_y):
		# Read X
		x = self.read_data('F:/Temp/aurora2/' + self.dataset + '/' + key_x)
		
		# Read Y
		y = self.read_data('F:/Temp/aurora2/' + self.dataset + '/' + key_y)

		return x, y

	def gen(self, indices):
		key_batch_x = [self.X[ix] for ix in indices]
		key_batch_y = [self.y[ix] for ix in indices]

		cur_batch_size = len(indices)

		X_batch = np.zeros((cur_batch_size, params.MAX_LENGTH), dtype=theano.config.floatX)
		y_batch = np.zeros((cur_batch_size, params.MAX_LENGTH), dtype=theano.config.floatX)

		# Read all images in the batch
		for i in range(len(key_batch_x)):
			X_batch[i], y_batch[i] = self.process(key_batch_x[i], key_batch_y[i])

		# Transform the batch (augmentation, fft, normalization, etc.)
		X_batch, y_batch = self.transform(X_batch, y_batch)

		return X_batch, y_batch

	def __iter__(self):
		queue = JoinableQueue(maxsize=params.N_PRODUCERS * 8)

		n_batches, job_queue = self.start_producers(queue)

		# Run as consumer (read items from queue, in current thread)
		for x in xrange(n_batches):
			item = queue.get()
			yield item
			queue.task_done()

		queue.close()
		job_queue.close()

	def start_producers(self, result_queue):
		jobs = Queue()
		n_workers = params.N_PRODUCERS
		batch_count = 0

		# Flag used for keeping values in queue in order
		last_queued_job = Value('i', -1)

		for job_index, batch in enumerate(self.chunks(range(0, len(self.X)), self.batch_size)):
			batch_count += 1
			jobs.put((job_index, batch))

		# Define producer (putting items into queue)
		def produce(id):
			while True:
				job_index, task = jobs.get()

				if task is None:
					break

				result = self.gen(task)

				while(True):
					# My turn to add job done
					if last_queued_job.value == job_index - 1:
						with last_queued_job.get_lock():
							result_queue.put(result)
							last_queued_job.value += 1
							break

		# Start workers
		for i in xrange(n_workers):
			if params.MULTIPROCESS:
				p = Process(target=produce, args=(i,))
			else:
				p = Thread(target=produce, args=(i,))

			p.daemon = True
			p.start()

		# Add poison pills to queue (to signal workers to stop)
		for i in xrange(n_workers):
			jobs.put((-1, None))

		return batch_count, jobs

	def transform(self, Xb, yb):
		# Do fft or something
		return Xb, yb
