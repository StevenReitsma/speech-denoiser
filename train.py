from neuralnet import *
from params import *
from random import shuffle
import pickle
import os

def create_file_list(folder_clean, single=False):
    y = []
    X = []
    for file in os.listdir('aurora2/train/' + folder_clean):
        filename_clean = folder_clean + '/' + file
        if file != '.ddf' and filename_clean[-4:] != '.npy':
            if single and (file.find('.')-file.find('_')>3):
                continue
            y += [filename_clean]
            X += [filename_clean]
            print('adding: ', filename_clean, filename_clean)
        for i in range(4):
            for j in range(4):
                # check if noisy version exists
                filename_noisy = 'multi/N' + str(i + 1) + '_SNR' + str((j + 1) * 5) + '/' + file
                if os.path.isfile('aurora2/train/' + filename_noisy) and file != '.ddf' and filename_noisy[-4:] != '.npy':
                    y += [filename_clean]
                    X += [filename_noisy]
                    print('adding: ', filename_clean, filename_noisy)
    return X, y

if __name__ == "__main__":
    # find all clean files, and add with all noisy files, INCLUDING CLEAN TO CLEAN
    np.random.seed(0)
    folder_clean = 'clean'
    try:
        with open('file_list.pickle', 'rb') as f:
            X_train, y_train = pickle.load(f)
    except:
        X_train, y_train = create_file_list(folder_clean, single=True)
        with open('file_list.pickle', 'wb') as f:
            pickle.dump((X_train, y_train), f)

    shuffled_idx = np.random.permutation(len(X_train))
    X_new = []
    y_new = []
    for i in range(len(X_train)):
        X_new += [X_train[shuffled_idx[i]]]
        y_new += [y_train[shuffled_idx[i]]]
    X_train = X_new
    y_train = y_new
    print('Data size: ', len(X_train))
    # 1/3 validation
    n_train = int((len(X_train)/3.0)*2)
    X_valid = X_train[n_train:]
    y_valid = y_train[n_train:]
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]

    net = NeuralNetwork(params.BATCH_SIZE, X_train, X_valid, y_train, y_valid, preprocess=False)
    net.fit()
