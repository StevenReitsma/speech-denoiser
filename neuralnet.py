from sklearn.preprocessing import StandardScaler

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

from model import build_model, build_model_small, build_model_dense
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

    def __init__(self, batch_size, X_train, X_valid, y_train, y_valid, preprocess=False):
        self.max_epochs = par.EPOCHS
        self.learning_rate = par.START_LEARNING_RATE
        self.update_learning_rate = theano.shared(np.float32(self.learning_rate))
        self.batch_size = batch_size
        self.n_batches = ceil(len(X_train) / float(batch_size))
        self.n_val_batches = ceil(len(X_valid) / float(batch_size))

        self.batch_iterator_train = ParallelBatchIterator(X_train, y_train, batch_size, 'train', shuffle=True, preprocess=preprocess)
        self.batch_iterator_test = ParallelBatchIterator(X_valid, y_valid, batch_size, 'train', preprocess=preprocess)
        self.batch_iterator_total = ParallelBatchIterator(X_train+X_valid, y_train+y_valid, batch_size, 'train')

        self.logger = PrintLog()
        self.create_iterator_functions()

    def float16(self, X):
        return [x.astype(np.float16) for x in X]

    def create_iterator_functions(self):
        # Define input and target variables
        input_var = T.ftensor3('inputs')
        target_var = T.ftensor3('targets')
        hop_length = (par.STEP_SIZE / 1000.0) * par.SR
        self.net = build_model_small((None, par.N_COMPONENTS, int(par.MAX_LENGTH/hop_length)), input_var)

        # Define prediction and loss calculation
        prediction = lasagne.layers.get_output(self.net['prob'], inputs=input_var)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        # Define updates
        params = lasagne.layers.get_all_params(self.net['prob'], trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=self.update_learning_rate, momentum=0.9)

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
        standard_scaler = StandardScaler(copy=False)
        # train standardizer
        for Xb, yb, filename in tqdm(self.batch_iterator_train, total=self.n_batches):
            standard_scaler.partial_fit(yb.reshape(Xb.shape[0], -1))

        for epoch in range(0, self.max_epochs):
            t0 = time()

            train_losses = []
            valid_losses = []

            for Xb, yb, filename in tqdm(self.batch_iterator_train, total=self.n_batches):
                Xb = standard_scaler.transform(Xb.reshape(Xb.shape[0], -1)).reshape(Xb.shape)
                yb = standard_scaler.transform(yb.reshape(Xb.shape[0], -1)).reshape(Xb.shape)
                loss = self.train_fn(Xb, yb)
                train_losses.append(loss)

            for Xb, yb, filename in tqdm(self.batch_iterator_test, total=self.n_val_batches):
                Xb = standard_scaler.transform(Xb.reshape(Xb.shape[0], -1)).reshape(Xb.shape)
                yb = standard_scaler.transform(yb.reshape(Xb.shape[0], -1)).reshape(Xb.shape)
                loss, prediction = self.val_fn(Xb, yb)
                valid_losses.append(loss)

            # visualize sample
            for j in range(10):
                plt.clf()
                plt.imshow(np.concatenate((Xb[j], np.ones((Xb.shape[1], 1)), yb[j], np.ones((Xb.shape[1], 1)), prediction[j]), axis=1), aspect='auto')
                plt.axis('off')
                plt.title('real/ target/ reconstruction')
                plt.savefig('visualizations/' + 'sample_'+str(j)+'.png')

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)


            if avg_train_loss > best_train_loss * 0.999:
                self.update_learning_rate.set_value(self.update_learning_rate.get_value() * np.float32(0.99))
                print('new learning rate: ', self.update_learning_rate.get_value())
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

        print('Saving denoised files to disk')
        for Xb, yb, filename in tqdm(self.batch_iterator_total, total=self.n_batches):
            loss, prediction = self.val_fn(Xb, yb)
            for j in range(Xb.shape[0]):
                with open('aurora2/train_denoised' + '/'+filename[j]+'.npy', 'wb') as f:
                    np.save(f, prediction[j])

    def print_progress(self, train_history):
        self.logger(train_history)
        # plot progress:
        train_losses = []
        valid_losses = []
        for e in range(len(train_history)):
            train_losses +=[train_history[e]['train_loss']]
            valid_losses += [train_history[e]['valid_loss']]
        plt.clf()
        plt.plot(np.arange(len(train_history)), train_losses, label='Training loss')
        plt.plot(np.arange(len(train_history)), valid_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Reconstruction error convergence')
        plt.savefig('visualizations/convergence_plot.png')
