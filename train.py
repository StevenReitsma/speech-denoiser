from neuralnet import *
from params import *
from random import shuffle
import os

def create_file_list(folder_clean):
    y = []
    X = []
    for file in os.listdir('aurora2/train/' + folder_clean):
        filename_clean = folder_clean + '/' + file
        y += [filename_clean]
        X += [filename_clean]
        print('adding: ', filename_clean, filename_clean)
        for i in range(4):
            for j in range(4):
                # check if noisy version exists
                filename_noisy = 'multi/N' + str(i + 1) + '_SNR' + str((j + 1) * 5) + '/' + file
                if os.path.isfile('aurora2/train/' + filename_noisy) and file != '.ddf':
                    y += [filename_clean]
                    X += [filename_noisy]
                    print('adding: ', filename_clean, filename_noisy)
    return X, y

if __name__ == "__main__":
    X_train = ['clean/FAC_1A.08', 'clean/FAC_2A.08', 'clean/FAC_3A.08', 'clean/FAC_4A.08', 'clean/FAC_5A.08',
               'clean/FAC_6A.08', 'clean/FAC_7A.08', 'clean/FAC_8A.08', 'clean/FAC_9A.08']  # etc
    y_train = ['multi/N1_SNR20/FAC_1A.08', 'multi/N1_SNR10/FAC_2A.08', 'multi/N4_SNR20/FAC_3A.08',
               'multi/N3_SNR5/FAC_4A.08', 'multi/N4_SNR5/FAC_5A.08', 'multi/N3_SNR15/FAC_6A.08',
               'multi/N3_SNR5/FAC_7A.08', 'multi/N1_SNR10/FAC_8A.08', 'multi/N2_SNR5/FAC_9A.08']

    # find all clean files, and add with all noisy files, INCLUDING CLEAN TO CLEAN
    folder_clean = 'clean'
    X_train, y_train = create_file_list(folder_clean)

    X_train = X_train[:]
    y_train = y_train[:]
    print('Data size: ', len(X_train))
    # TODO: Use different validation set and test set.
    X_valid = X_train
    y_valid = y_train

    net = NeuralNetwork(params.BATCH_SIZE, X_train, X_valid, y_train, y_valid)
    net.fit()
