import os
import numpy as np
from barkgram import *
from Sample import Sample
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








# Create BFD Dataset

print('BFD Aug Dataset')

Dataset_Str = 'BFD'

path_audio = '../../data/external/BFD_Dataset'

list_wav = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

num_cuts = 20

cut_step = len(list_wav)//num_cuts

for cut in range(num_cuts):

    start = cut*cut_step
    end = (cut+1)*cut_step

    if cut!=num_cuts-1:
        list_wav_cut = list_wav[start:end]
    else:
        list_wav_cut = list_wav[start:]

    Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
    Classes = []

    print([start,end])
    print(list_wav_cut)

    for i in range(len(list_wav_cut)):

        print(str(i) + ' of ' + str(len(list_wav_cut)))

        audio, fs = sf.read(list_wav_cut[i])
        audio_ref = audio/np.max(abs(audio))

        if '/Kick/' in list_wav_cut[i]:
            Class = 'kd'
        elif '/Snare/' in list_wav_cut[i]:
            Class = 'sd'
        elif '/Closed_HiHat/' in list_wav_cut[i]:
            Class = 'hhc'
        elif '/Opened_HiHat/' in list_wav_cut[i]:
            Class = 'hho'
        
        for k in range(7):

            kn = np.random.randint(0,2)
            pt = np.random.uniform(low=-1, high=1, size=None)
            st = np.random.uniform(low=0.8, high=1.2, size=None)

            if k!=0:
                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)
            else:
                audio = audio_ref

            Spec = Sample(audio,n_fft=4096,hop_size=512,n_bands=128,bark_basis=weights,bark_freqs=cent_freqs,ear_basis=db_diffs,numpy_input=True).bgram

            Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec,axis=0)))
            Classes.append(Class)

    Spec_Matrix_All = Spec_Matrix_All[1:]

    np.save('../../data/interim/Dataset_BFD_' + str(cut), Spec_Matrix_All)
    np.save('../../data/interim/Classes_BFD_' + str(cut), np.array(Classes))

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('../../data/interim/Dataset_BFD_' + str(cut) + '.npy')))
    Classes_All = np.concatenate((Classes_All,np.load('../../data/interim/Classes_BFD_' + str(cut) + '.npy')))
    os.remove('../../data/interim/Dataset_BFD_' + str(cut) + '.npy')
    os.remove('../../data/interim/Classes_BFD_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('../../data/interim/Dataset_BFD_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_BFD_Bark', Classes_All)

    




