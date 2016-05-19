# speech-denoiser

#TODO: track reconstruction errors per noise level

#TODO: add noise, to generate more training samples
#TODO: segment longer sequences to get more samples
#TODO: more than 1second or sample from each audio clip instead of taking middle
#TODO: add delta and delta/delta mfcc to improve classification
#TODO: mel denoising
#TODO: 2d mel denoising
#TODO: dense layer+maxpooling layers to create more invariance.
#TODO: batch norm, dropout, bigger layers

#TODO: classifier improvements, optimize network architecture

Some files are longer than 10000 samples, taken middle.

('Data size: ', 15192) includes multi digits

('Data size: ', ~5000?) singles

results:

validation accuracy: 0.85615

(' Denoised clean accuracy: ', 0.85696517412935325)

(' Denoised multi accuracy: ', 0.67873771339886191)

(' Denoised multi accuracy for SNR5: ', 0.53361344537815125)
(' Denoised multi accuracy for SNR10: ', 0.63080168776371304)
(' Denoised multi accuracy for SNR15: ', 0.71721311475409832)
(' Denoised multi accuracy for SNR20: ', 0.78585858585858581)


(' Noisy multi accuracy: ', 0.25504397309881016)

' Noisy multi accuracy for SNR5: ', 0.19327731092436976)
(' Noisy multi accuracy for SNR10: ', 0.22151898734177214)
(' Noisy multi accuracy for SNR15: ', 0.25614754098360654)
(' Noisy multi accuracy for SNR20: ', 0.35555555555555557)