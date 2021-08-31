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

frame_sizes = [512,1024,2048]
num_specs = [128]
num_frames = 128
hop_size = 345
delta_bool = False


# Create Misc Dataset

Dataset_Str = 'Misc'

path_audio = '../../data/external/Misc_Dataset'

list_wav = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

for w in enumerate(num_specs):
    
    for j in enumerate(frame_sizes):

        frame_size = frame_sizes[j]
        num_spec = num_specs[w]

        Spec_Matrix_All = np.zeros((1,num_frames,num_spec))
        Classes = []

        for i in enumerate(list_wav):

            audio, fs = librosa.load(list_wav[i], sr=44100)
            if len(audio)<frame_size:
                continue
            audio_ref = audio/np.max(abs(audio))

            if '/Kick/' in list_wav[i]:
                Class = 'Kick'
            elif '/Snare/' in list_wav[i]:
                Class = 'Snare'
            elif '/HiHat/' in list_wav[i]:
                Class = 'HH'
            else:
                print('No class')
            
            for k in range(10):

                kn = np.random.randint(0,2)
                pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                st = np.random.uniform(low=0.8, high=1.2, size=None)

                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)

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

        np.save('../../data/interim/Misc/Dataset_' + Dataset_Str + '_' + str(frame_size), Spec_Matrix_All)
        np.save('../../data/interim/Misc/Classes_' + Dataset_Str + '_' + str(frame_size), np.array(Classes))




# Create Misc Dataset

Dataset_Str = 'BFD'

path_audio = '../../data/external/BFD_Dataset'

list_wav = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

for w in enumerate(num_specs):
    
    for j in enumerate(frame_sizes):

        frame_size = frame_sizes[j]
        num_spec = num_specs[w]

        Spec_Matrix_All = np.zeros((1,num_frames,num_spec))
        Classes = []

        for i in enumerate(list_wav):

            audio, fs = librosa.load(list_wav[i], sr=44100)
            if len(audio)<frame_size:
                continue
            audio_ref = audio/np.max(abs(audio))

            if '/Kick/' in list_wav[i]:
                Class = 'Kick'
            elif '/Snare/' in list_wav[i]:
                Class = 'Snare'
            elif '/HiHat/' in list_wav[i]:
                Class = 'HH'
            else:
                print('No class')
            
            for k in range(10):

                kn = np.random.randint(0,2)
                pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                st = np.random.uniform(low=0.8, high=1.2, size=None)

                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)

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

        np.save('../../data/interim/BFD/Dataset_' + Dataset_Str + '_' + str(frame_size), Spec_Matrix_All)
        np.save('../../data/interim/BFD/Classes_' + Dataset_Str + '_' + str(frame_size), np.array(Classes))




        
