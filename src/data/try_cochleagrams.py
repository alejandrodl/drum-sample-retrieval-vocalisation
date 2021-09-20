import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pycochleagram.cochleagram as cgram

signal, fs = sf.read('/Users/alejandrodelgadoluezas/Documents/GitHub/vocal-percussion-transcription/data/external/LVT_Audio/Dataset_Train_00.wav')

sr = 44100
num_filters = 96
sample_factor = 2
downsample = 2
nonlinearity = 'db'

cochleagram = cgram.human_cochleagram(signal, sr, n=num_filters, sample_factor=sample_factor, downsample=downsample, nonlinearity=nonlinearity, strict=False)
cochleagram = np.flipud(cochleagram)

plt.figure()
plt.title('Signal waveform')
plt.plot(signal)
plt.show()

plt.figure()
plt.title('Cochleagram with nonlinearity: "log"')
plt.imshow(cochleagram)
plt.show()