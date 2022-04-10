import os
import pdb
import numpy as np
import librosa

import essentia
from essentia.standard import *


class extract_heuristic():
    
    def __init__(self):
        
        self.num_mel_coeffs = 12
        self.num_mel_bands = 40
        self.num_features = 32
        
    def compute(self, audio, framesize):
        
        hopsize = 512
        audio = essentia.array(audio)
        feature_vector = np.zeros(self.num_features)
        
        # Define preprocessing functions
        
        win = Windowing(type='hann')
        spec = Spectrum(size=framesize)
        
        # Define extractors
        
        mfcc = MFCC(highFrequencyBound=22050,numberCoefficients=self.num_mel_coeffs+1,numberBands=self.num_mel_bands,inputSize=(framesize//2)+1)
        
        env = Envelope(applyRectification=True, attackTime=5, releaseTime=100)
        der = DerivativeSFX()
        flat = FlatnessSFX()
        tct = TCToTotal()
        loud = Loudness()
        ptch = PitchYin()
        cent = SpectralCentroidTime()
        #dur = EffectiveDuration(thresholdRatio=0.2)
        
        # Extract and allocate envelope features
        
        feature_vector[0] = len(audio)/44100

        envelope = env(audio)
        feature_vector[1], _ = der(envelope)

        # Extract MFCC

        _mfccs = []
        _loud = []
        _ptch = []
        _cent = []
        for frame in FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
            _, _mfccv = mfcc(spec(win(frame)))
            _mfccs.append(_mfccv)
            _loudv = loud(frame)
            _loud.append(_loudv)
            _ptchv, _ = ptch(frame)
            _ptch.append(_ptchv)
            _centv = cent(frame)
            _cent.append(_centv)
            
        # Allocate MFCC

        feature_vector[2] = np.mean(np.array(_loud))
        feature_vector[3] = np.std(np.array(_loud))
        feature_vector[4] = np.mean(np.array(_ptch))
        feature_vector[5] = np.std(np.array(_ptch))
        feature_vector[6] = np.mean(np.array(_cent))
        feature_vector[7] = np.std(np.array(_cent))
        for i in range(self.num_mel_coeffs):
            feature_vector[8+i] = np.mean(np.array(_mfccs)[:,i+1])
        for i in range(self.num_mel_coeffs):
            feature_vector[8+self.num_mel_coeffs+i] = np.mean(np.gradient(np.array(_mfccs)[:,i+1]))

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

extractor = extract_heuristic()
features = np.zeros((len(list_wav),extractor.num_features))

for n in range(len(list_wav)):
    audio, fs = librosa.load(list_wav[n], sr=44100)
    audio = audio/np.max(abs(audio))
    features[n] = extractor.compute(audio,frame_size)
print(features.shape)

np.save('../../data/processed/' + mode + '/Dataset_Ref_Heuristic', features)


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

extractor = extract_heuristic()
features = np.zeros((len(list_wav),extractor.num_features))

for n in range(len(list_wav)):
    audio, fs = librosa.load(list_wav[n], sr=44100)
    audio = audio/np.max(abs(audio))
    features[n] = extractor.compute(audio,frame_size)
print(features.shape)

np.save('../../data/processed/' + mode + '/Dataset_Imi_Heuristic', features)