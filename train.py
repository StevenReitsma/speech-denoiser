from neuralnet import *
from params import *
from random import shuffle


if __name__ == "__main__":
	X_train = ['clean/FAC_1A.08', 'clean/FAC_2A.08', 'clean/FAC_3A.08', 'clean/FAC_4A.08', 'clean/FAC_5A.08', 'clean/FAC_6A.08', 'clean/FAC_7A.08', 'clean/FAC_8A.08', 'clean/FAC_9A.08'] #etc
	y_train = ['multi/N1_SNR20/FAC_1A.08', 'multi/N1_SNR10/FAC_2A.08', 'multi/N4_SNR20/FAC_3A.08', 'multi/N3_SNR5/FAC_4A.08', 'multi/N4_SNR5/FAC_5A.08', 'multi/N3_SNR15/FAC_6A.08', 'multi/N3_SNR5/FAC_7A.08', 'multi/N1_SNR10/FAC_8A.08', 'multi/N2_SNR5/FAC_9A.08']

	# Temporary! Use different validation set, obviously.
	X_valid = X_train
	y_train = y_valid

	net = NeuralNetwork(params.BATCH_SIZE, X_train, X_valid, y_train, y_valid)
	net.fit()
