import os
import numpy as np
from barkgram import *
from Sample import Sample
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


# Helpers

freqs = calc_bark_spaced_cent_freqs(n_bands=128)
freqs = freqs[1:-1]

db_diffs = ear_model_basis(freqs=freqs)
cent_freqs = calc_bark_spaced_cent_freqs(n_bands=128)
weights =  bark_basis(fs=44100, n_fft=4096, n_bands=128)

# Reference drum sounds

path_audio = '../../data/external/VIPS_Dataset_KSH/drum_sounds'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

dataset_adib_ref = np.zeros((len(list_wav),128,128))
for n in range(len(list_wav)):
    sample = Sample(list_wav[n], n_fft=4096, hop_size=512, n_bands=128, bark_basis=weights, bark_freqs=cent_freqs, ear_basis=db_diffs)
    dataset_adib_ref[n] = sample.bgram

np.save('../../data/interim/Dataset_VIPS_Ref_Bark', dataset_adib_ref)


# Vocal imitation sounds

path_audio = '../../data/external/VIPS_Dataset_KSH/imitations'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

dataset_adib_imi = np.zeros((len(list_wav),128,128))
for n in range(len(list_wav)):
    sample = Sample(list_wav[n], n_fft=4096, hop_size=512, n_bands=128, bark_basis=weights, bark_freqs=cent_freqs, ear_basis=db_diffs)
    dataset_adib_imi[n] = sample.bgram

np.save('../../data/interim/Dataset_VIPS_Imi_Bark', dataset_adib_imi)