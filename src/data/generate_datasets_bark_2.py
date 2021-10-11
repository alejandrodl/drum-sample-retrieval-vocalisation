import os
import numpy as np
from barkgram import *
from Sample import Sample
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pyrubberband as pyrb



# Data Augmentation Functions
    
def pitch_shift(data, sampling_rate, pitch_semitones):
    return pyrb.pitch_shift(data, sampling_rate, pitch_semitones)

def time_stretch(data, stretch_factor):
    return pyrb.time_stretch(data, 44100, stretch_factor)



# Helpers

freqs = calc_bark_spaced_cent_freqs(n_bands=128)
freqs = freqs[1:-1]

db_diffs = ear_model_basis(freqs=freqs)
cent_freqs = calc_bark_spaced_cent_freqs(n_bands=128)
weights =  bark_basis(fs=44100, n_fft=4096, n_bands=128)



num_spec = 128
num_frames = 128




# Create AVP Dataset

print('AVP Aug Dataset')

path_audio = '../../data/external/AVP_Dataset/Personal'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

num_cuts = 20

cut_step = len(list_wav)//num_cuts

for cut in range(num_cuts):

    start = cut*cut_step
    end = (cut+1)*cut_step

    if cut!=num_cuts-1:
        list_wav_cut = list_wav[start:end]
        list_csv_cut = list_csv[start:end]
    else:
        list_wav_cut = list_wav[start:]
        list_csv_cut = list_csv[start:]

    Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
    Classes_All = np.zeros(1)

    print([start,end])
    print(list_wav_cut)

    for i in range(len(list_wav_cut)):

        onsets = np.loadtxt(list_csv_cut[i], delimiter=',', usecols=0)
        
        audio, fs = sf.read(list_wav_cut[i])
        audio_ref = audio/np.max(abs(audio))

        onsets_samples = onsets*fs
        onsets_ref = onsets_samples.astype(int)
        
        for k in range(9):

            Classes = np.loadtxt(list_csv_cut[i], delimiter=',', usecols=1, dtype=np.unicode_)

            kn = np.random.randint(0,2)
            pt = np.random.uniform(low=-1.5, high=1.5, size=None)
            st = np.random.uniform(low=0.8, high=1.2, size=None)

            if k!=0:
                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
            else:
                audio = audio_ref
                onsets = onsets_ref
        
            if len(onsets)!=len(Classes):
                print('Classes-Onsets mismatch')

            Spec_Matrix = np.zeros((len(onsets),128,128))
            for n in range(len(onsets)-1):
                Spec_Matrix[n] = Sample(audio[onsets[n]:onsets[n+1]],n_fft=4096,hop_size=512,n_bands=128,bark_basis=weights,bark_freqs=cent_freqs,ear_basis=db_diffs,numpy_input=True).bgram
            Spec_Matrix[n+1] = Sample(audio[onsets[n+1]:],n_fft=4096,hop_size=512,n_bands=128,bark_basis=weights,bark_freqs=cent_freqs,ear_basis=db_diffs,numpy_input=True).bgram
        
            Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
            Classes_All = np.concatenate((Classes_All,Classes))
        
        Spec_Matrix_All = Spec_Matrix_All[1:]
        Classes_All = Classes_All[1:]

    np.save('../../data/interim/Dataset_AVP_' + str(cut), Spec_Matrix_All)
    np.save('../../data/interim/Classes_AVP_' + str(cut), Classes_All)

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('../../data/interim/Dataset_AVP_' + str(cut) + '.npy')))
    Classes_All = np.concatenate((Classes_All,np.load('../../data/interim/Classes_AVP_' + str(cut) + '.npy')))
    os.remove('../../data/interim/Dataset_AVP_' + str(cut) + '.npy')
    os.remove('../../data/interim/Classes_AVP_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('../../data/interim/Dataset_AVP_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_AVP_Bark', Classes_All)




