#!/usr/bin/env python
# coding: utf-8

import os
import librosa
import numpy as np
import pyrubberband as pyrb



def pitch_shift(data, sampling_rate, pitch_semitones):
    return pyrb.pitch_shift(data, sampling_rate, pitch_semitones)

def time_stretch(data, stretch_factor):
    return pyrb.time_stretch(data, 44100, stretch_factor)


# Params

frame_size = 1024
num_spec = 128
num_frames = 128
hop_size = 345
delta_bool = False



# Create BFD Dataset

print('BFD Aug Dataset')

Dataset_Str = 'BFD'

path_audio = 'data/external/BFD_Dataset'

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

        audio, fs = librosa.load(list_wav_cut[i], sr=44100)
        if len(audio)<frame_size:
            continue
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

            Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

            location = 0

            Spec = Dataset_Spec[location:]
            if Spec.shape[0]<num_frames:
                Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
            elif Spec.shape[0]>=num_frames:
                Spec = Spec[:num_frames]

            Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))
            Classes.append(Class)

    Spec_Matrix_All = Spec_Matrix_All[1:]

    np.save('data/interim/Dataset_BFD_' + str(cut), Spec_Matrix_All)
    np.save('data/interim/Classes_BFD_' + str(cut), np.array(Classes))

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('data/interim/Dataset_BFD_' + str(cut) + '.npy')))
    Classes_All = np.concatenate((Classes_All,np.load('data/interim/Classes_BFD_' + str(cut) + '.npy')))
    os.remove('data/interim/Dataset_BFD_' + str(cut) + '.npy')
    os.remove('data/interim/Classes_BFD_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_BFD', Spec_Matrix_All)
np.save('data/interim/Classes_BFD', Classes_All)



# Create Misc Aug Dataset

print('Misc Aug Dataset')

Dataset_Str = 'Misc'

path_audio = 'data/external/Misc_Dataset'

list_wav = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

num_cuts = 80

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

        audio, fs = librosa.load(list_wav_cut[i], sr=44100)
        if len(audio)<frame_size:
            continue
        audio_ref = audio/np.max(abs(audio))

        if '/Kick/' in list_wav[i]:
            Class = 'kd'
        elif '/Snare/' in list_wav[i]:
            Class = 'sd'
        elif '/Closed_HiHat/' in list_wav[i]:
            Class = 'hhc'
        elif '/Opened_HiHat/' in list_wav[i]:
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

            Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

            location = 0

            Spec = Dataset_Spec[location:]
            if Spec.shape[0]<num_frames:
                Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
            elif Spec.shape[0]>=num_frames:
                Spec = Spec[:num_frames]

            Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))
            Classes.append(Class)

    Spec_Matrix_All = Spec_Matrix_All[1:]

    np.save('data/interim/Dataset_Misc_' + str(cut), Spec_Matrix_All)
    np.save('data/interim/Classes_Misc_' + str(cut), np.array(Classes))

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('data/interim/Dataset_Misc_' + str(cut) + '.npy')))
    Classes_All = np.concatenate((Classes_All,np.load('data/interim/Classes_Misc_' + str(cut) + '.npy')))
    os.remove('data/interim/Dataset_Misc_' + str(cut) + '.npy')
    os.remove('data/interim/Classes_Misc_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_Misc', Spec_Matrix_All)
np.save('data/interim/Classes_Misc', Classes_All)
    



