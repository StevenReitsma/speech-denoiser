from iterators import ParallelBatchIterator
import pickle
import lasagne
import numpy as np
from time import time
import theano
import theano.tensor as T
from tabulate import tabulate
from collections import OrderedDict
import sys

from model import build_model
from params import params as par
from lasagne.layers import DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.regularization import regularize_network_params, l2
from tqdm import *
from math import ceil
import matplotlib.pyplot as plt
import colorama
colorama.init()


class ansi:
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog:
    def __init__(self):
        self.first_iteration = True

    def __call__(self, train_history):
        print(self.table(train_history))
        sys.stdout.flush()

    def table(self, train_history):
        info = train_history[-1]

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.CYAN if info['train_loss_best'] else "",
                info['train_loss'],
                ansi.ENDC if info['train_loss_best'] else "",
                )),
            ('valid loss', "{}{:.5f}{}".format(
                ansi.GREEN if info['valid_loss_best'] else "",
                info['valid_loss'],
                ansi.ENDC if info['valid_loss_best'] else "",
                )),
            ('valid acc', info['valid_accuracy']),
            ('train/val', info['train_loss'] / info['valid_loss']),
            ])

        info_tabulate['duration'] = "{:.2f}s".format(info['duration'])

        tabulated = tabulate(
            [info_tabulate], headers="keys", floatfmt='.5f')

        out = ""
        if self.first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"
            self.first_iteration = False

        out += tabulated.rsplit('\n', 1)[-1]
        return out


class NeuralNetwork():

    def __init__(self, batch_size, X_train, X_valid, y_train, y_valid):
        self.max_epochs = par.EPOCHS
        self.learning_rate = par.START_LEARNING_RATE
        self.batch_size = batch_size
        self.n_batches = ceil(len(X_train) / float(batch_size))
        self.n_val_batches = ceil(len(X_valid) / float(batch_size))

        self.n_components = 128
        self.mfcc = False

        self.batch_iterator_train = ParallelBatchIterator(X_train, y_train, batch_size, 'train', self.n_components, self.mfcc)
        self.batch_iterator_test = ParallelBatchIterator(X_valid, y_valid, batch_size, 'train', self.n_components, self.mfcc)

        self.logger = PrintLog()
        self.create_iterator_functions()

    def float16(self, X):
        return [x.astype(np.float16) for x in X]

    def create_iterator_functions(self):
        # Define input and target variables
        input_var = T.ftensor3('inputs')
        target_var = T.ftensor3('targets')
        self.net = build_model((None, self.n_components, par.MAX_LENGTH/500), input_var)

        # Define prediction and loss calculation
        prediction = lasagne.layers.get_output(self.net['prob'], inputs=input_var)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        # Define updates
        params = lasagne.layers.get_all_params(self.net['prob'], trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=self.learning_rate, momentum=0.9)

        # Define test time prediction
        test_prediction = lasagne.layers.get_output(self.net['prob'], inputs=input_var, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()

        # Compile functions
        self.train_fn = theano.function([input_var, target_var], loss, updates=updates)
        self.val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])

    def fit(self):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        train_history = []
        for epoch in range(0, self.max_epochs):
            t0 = time()

            train_losses = []
            valid_losses = []

            for Xb, yb in tqdm(self.batch_iterator_train, total=self.n_batches):
                loss = self.train_fn(Xb, yb)
                train_losses.append(loss)

            for Xb, yb in tqdm(self.batch_iterator_test, total=self.n_val_batches):
                loss, prediction = self.val_fn(Xb, yb)
                valid_losses.append(loss)

            # visualize sample
            plt.imshow(np.concatenate((Xb[0], np.ones((Xb.shape[1], 1)), yb[0], np.ones((Xb.shape[1], 1)), prediction[0]), axis=1))
            plt.axis('off')
            plt.title('real/target/reconstruction')
            plt.savefig('visualizations/' + 'sample.png')

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss,
                'valid_accuracy': 'N/A',
                'duration': time() - t0,
            }

            train_history.append(info)

            self.print_progress(train_history)

            # Save to disk
            vals = lasagne.layers.get_all_param_values(self.net['prob'])
            with open('models/' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(vals, f, -1)

    def print_progress(self, train_history):
        self.logger(train_history)
