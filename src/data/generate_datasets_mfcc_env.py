import os
import pdb
import numpy as np

import essentia
from essentia.standard import *


class extract_features_16():
    
    def __init__(self):
        
        self.num_extra_features = 4
        self.num_mel_coeffs = 12
        self.num_mel_bands = 40
        self.num_features = 16
        
    def compute(self, audio, framesize):
        
        hopsize = 345
        audio = essentia.array(audio)
        feature_vector = np.zeros(self.num_features)
        
        # Define preprocessing functions
        
        win = Windowing(type='hann')
        spec = Spectrum(size=framesize)
        
        # Define extractors
        
        mfcc = MFCC(highFrequencyBound=22050,numberCoefficients=self.num_mel_coeffs,numberBands=self.num_mel_bands,inputSize=(framesize//2)+1)
        
        env = Envelope(applyRectification=True, attackTime=5, releaseTime=100)
        der = DerivativeSFX()
        flat = FlatnessSFX()
        tct = TCToTotal()
        
        # Extract and allocate envelope features
        
        envelope = env(audio)
        feature_vector[0], feature_vector[1] = der(envelope)
        feature_vector[2] = flat(envelope)
        feature_vector[3] = tct(envelope)

        # Extract MFCC

        _mfccs = []
        for frame in FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
            _, _mfccv = mfcc(spec(win(frame)))
            _mfccs.append(_mfccv)
            
        # Allocate MFCC
            
        for i in range(self.num_mel_coeffs):
            feature_vector[4+i] = np.mean(np.array(_mfccs)[:,i])

        return feature_vector
    
    
mode = 'eng_mfcc_env'
frame_size = 2048

if not os.path.isdir('../../data/processed/' + mode):
    os.mkdir('../../data/processed/' + mode)

# Create VIPS Dataset Reference

print('VIPS Dataset Reference')

path_audio = '../../data/external/VIPS_Dataset_KSH/drum_sounds'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

extractor = extract_features_16()
features = np.zeros((len(list_wav),extractor.num_features))

for i in range(len(list_wav)):

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    features[n] = extractor.compute(audio,frame_size)

np.save('../../data/processed/' + mode + '/Dataset_VIPS_Ref_MFCC_ENV', Spec_Matrix_All)



# Create VIPS Dataset Imitations

print('VIPS Dataset Imitations')

path_audio = '../../data/external/VIPS_Dataset_KSH/imitations'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

extractor = extract_features_16()
features = np.zeros((len(list_wav),extractor.num_features))

for i in range(len(list_wav)):

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    features[n] = extractor.compute(audio,frame_size)

np.save('../../data/processed/' + mode + '/Dataset_VIPS_Imi_MFCC_ENV', Spec_Matrix_All)