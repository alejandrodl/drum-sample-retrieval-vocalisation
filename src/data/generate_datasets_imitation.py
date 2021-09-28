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



frame_size = 1024
num_spec = 128
num_frames = 128

hop_size = 345
delta_bool = False



# Create AVP Dataset

print('AVP Aug Dataset')

path_audio = 'data/external/AVP_Dataset/Personal'

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

num_cuts = 50

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
        
        audio, fs = librosa.load(list_wav_cut[i], sr=44100)
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
        
            spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

            if delta_bool:
                delta = librosa.feature.delta(spec)
                Dataset_Spec = np.concatenate((spec, delta), axis=1)
            else:
                Dataset_Spec = spec

            Onsets = np.zeros(spec.shape[0])
            location = np.floor(onsets/hop_size)
            if (location.astype(int)[-1]<len(Onsets)):
                Onsets[location.astype(int)] = 1
            else:
                Onsets[location.astype(int)[:-1]] = 1

            num_onsets = int(np.sum(Onsets))
            if num_onsets!=len(Classes):
                raise('num_onsets==len(Classes)')
            Spec_Matrix = np.zeros((num_onsets,num_frames,num_spec))

            L = len(Onsets)
            count = 0
            for n in range(L):
                if Onsets[n]==1:
                    c = 1
                    while Onsets[n+c]==0 and (n+c)<L-1:
                        c += 1
                    Spec = Dataset_Spec[n:n+c]
                    if c<num_frames:
                        Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                    elif c>=num_frames:
                        Spec = Spec[:num_frames]
                    Spec_Matrix[count] = Spec.T
                    count += 1
                    
            Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
            Classes_All = np.concatenate((Classes_All,Classes))
        
        Spec_Matrix_All = Spec_Matrix_All[1:]
        Classes_All = Classes_All[1:]

    np.save('data/interim/Dataset_AVP_' + str(cut), Spec_Matrix_All)
    np.save('data/interim/Classes_AVP_' + str(cut), Classes_All)

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('data/interim/Dataset_AVP_' + str(cut) + '.npy')))
    Classes_All = np.concatenate((Classes_All,np.load('data/interim/Classes_AVP_' + str(cut) + '.npy')))
    os.remove('data/interim/Dataset_AVP_' + str(cut) + '.npy')
    os.remove('data/interim/Classes_AVP_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_AVP', Spec_Matrix_All)
np.save('data/interim/Classes_AVP', Classes_All)





# Create AVP_Fixed_Small Aug Dataset

print('AVP_Fixed_Small Aug Dataset')

Dataset_Str = 'AVP_Fixed_Small_Aug'

path_audio = 'data/external/AVP_Dataset/Fixed'

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

list_wav = list_wav[2::5]
list_csv = list_csv[2::5]

list_wav = list_wav[::4]
list_csv = list_csv[::4]

Classes_All = np.zeros(1)
Spec_Matrix_All = np.zeros((1,num_spec,num_frames))

for i in range(len(list_wav)):

    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)
    
    for k in range(9):

        Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)

        for cl in range(len(Classes)):
            if Classes[cl]=='  ':
                Classes[cl]=='hhc'
            elif Classes[cl]=='0.0':
                Classes[cl]=='kd'
            elif Classes[cl]=='hhc ':
                Classes[cl]=='hhc'

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

        Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

        Onsets = np.zeros(Dataset_Spec.shape[0])
        location = np.floor(onsets/hop_size)
        if (location.astype(int)[-1]<len(Onsets)):
            Onsets[location.astype(int)] = 1
        else:
            Onsets[location.astype(int)[:-1]] = 1

        num_onsets = int(np.sum(Onsets))
        Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

        L = len(Onsets)
        count = 0
        for n in range(L):
            if Onsets[n]==1:
                c = 1
                while (n+c)<L-1 and Onsets[n+c]==0:
                    c += 1
                Spec = Dataset_Spec[n:n+c]
                if c<num_frames:
                    Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                elif c>=num_frames:
                    Spec = Spec[:num_frames]
                Spec_Matrix[count] = Spec.T
                count += 1

        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
        Classes_All = np.concatenate((Classes_All,Classes))

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_AVP_Fixed', Spec_Matrix_All)
np.save('data/interim/Classes_AVP_Fixed', Classes_All)







# Create LVT_2 Aug Dataset

print('LVT_2 Aug Dataset')

Dataset_Str = 'LVT2'

path_audio = 'data/external/LVT_Dataset/DataSet_Wav_Annotation'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('2.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for i in range(len(list_wav)):

    print(str(i) + ' of ' + str(len(list_wav)))
    
    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)
    
    for k in range(9):

        Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)

        for n in range(len(Classes)):
            if Classes[n]=='Kick':
                Classes[n] = 'kd'
            elif Classes[n]=='Snare':
                Classes[n] = 'sd'
            elif Classes[n]=='HH':
                Classes[n] = 'hhc'
            else:
                print('No class')

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

        Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

        Onsets = np.zeros(Dataset_Spec.shape[0])
        location = np.floor(onsets/hop_size)
        if (location.astype(int)[-1]<len(Onsets)):
            Onsets[location.astype(int)] = 1
        else:
            Onsets[location.astype(int)[:-1]] = 1

        num_onsets = int(np.sum(Onsets))
        Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

        L = len(Onsets)
        count = 0
        for n in range(L):
            if Onsets[n]==1:
                c = 1
                while Onsets[n+c]==0 and (n+c)<L-1:
                    c += 1
                Spec = Dataset_Spec[n:n+c]
                if c<num_frames:
                    Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                elif c>=num_frames:
                    Spec = Spec[:num_frames]
                Spec_Matrix[count] = Spec.T
                count += 1
    
        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
        Classes_All = np.concatenate((Classes_All,Classes))

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_LVT_2', Spec_Matrix_All)
np.save('data/interim/Classes_LVT_2', Classes_All)







# Create LVT_3 Aug Dataset

print('LVT_3 Aug Dataset')

Dataset_Str = 'LVT3'

path_audio = 'data/external/LVT_Dataset/DataSet_Wav_Annotation'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('3.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes_All = np.zeros(1)

for i in range(len(list_wav)):

    print(str(i) + ' of ' + str(len(list_wav)))
    
    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)
    
    for k in range(9):

        Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)

        for n in range(len(Classes)):
            if Classes[n]=='Kick':
                Classes[n] = 'kd'
            elif Classes[n]=='Snare':
                Classes[n] = 'sd'
            elif Classes[n]=='HH':
                Classes[n] = 'hhc'
            else:
                print('No class')

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

        Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

        Onsets = np.zeros(Dataset_Spec.shape[0])
        location = np.floor(onsets/hop_size)
        if (location.astype(int)[-1]<len(Onsets)):
            Onsets[location.astype(int)] = 1
        else:
            Onsets[location.astype(int)[:-1]] = 1

        num_onsets = int(np.sum(Onsets))
        Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

        L = len(Onsets)
        count = 0
        for n in range(L):
            if Onsets[n]==1:
                c = 1
                while Onsets[n+c]==0 and (n+c)<L-1:
                    c += 1
                Spec = Dataset_Spec[n:n+c]
                if c<num_frames:
                    Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                elif c>=num_frames:
                    Spec = Spec[:num_frames]
                Spec_Matrix[count] = Spec.T
                count += 1
    
        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
        Classes_All = np.concatenate((Classes_All,Classes))

Spec_Matrix_All = Spec_Matrix_All[1:]
Classes_All = Classes_All[1:]

np.save('data/interim/Dataset_LVT_3', Spec_Matrix_All)
np.save('data/interim/Classes_LVT_3', Classes_All)








# Create VIM_Percussive Aug Dataset

print('VIM_Percussive Dataset')

Dataset_Str = 'VIM_Percussive_Aug'

path_audio = 'data/external/VIM_Percussive_Dataset'

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

num_cuts = 50

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

        print(str(i) + ' of ' + str(len(list_wav_cut)))

        onsets = np.loadtxt(list_csv_cut[i], delimiter=',', usecols=0)
        
        audio, fs = librosa.load(list_wav_cut[i], sr=44100)
        audio_ref = audio/np.max(abs(audio))

        onsets_samples = onsets*fs
        onsets_ref = onsets_samples.astype(int)
        
        for k in range(9):

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

            Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

            if onsets.ndim==0:

                location = int(np.floor(onsets/hop_size))

                Spec = Dataset_Spec[location:]
                if Spec.shape[0]<num_frames:
                    Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
                elif Spec.shape[0]>=num_frames:
                    Spec = Spec[:num_frames]

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))

            else:

                Onsets = np.zeros(Dataset_Spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while (n+c)<L-1 and Onsets[n+c]==0:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))

    Spec_Matrix_All = Spec_Matrix_All[1:]
    np.save('data/interim/Dataset_VIM_Percussive_' + str(cut), Spec_Matrix_All)

    print(Spec_Matrix_All.shape)
    print(cut)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))

for cut in range(num_cuts):
    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.load('data/interim/Dataset_VIM_Percussive_' + str(cut) + '.npy')))
    os.remove('data/interim/Dataset_VIM_Percussive_' + str(cut) + '.npy')

Spec_Matrix_All = Spec_Matrix_All[1:]
np.save('data/interim/Dataset_VIM_Percussive_', Spec_Matrix_All)






# Create Zhu Aug Dataset

print('Beatbox Dataset')

Dataset_Str = 'Zhu_Aug'

path_audio = 'data/external/Beatbox_Dataset'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes = []

for i in range(len(list_wav)):

    print(str(i) + ' of ' + str(len(list_wav)))

    audio, fs = librosa.load(list_wav[i], sr=44100)
    if len(audio)<frame_size:
        continue
    audio_ref = audio/np.max(abs(audio))

    if '/Kick/' in list_wav[i]:
        Class = 'kd'
    elif '/R/' in list_wav[i]:
        Class = 'sd'
    elif '/Pf/' in list_wav[i]:
        Class = 'sd'
    elif '/Hihat/' in list_wav[i]:
        Class = 'hhc'
    else:
        print('No class')
    
    for k in range(9):

        kn = np.random.randint(0,2)
        pt = np.random.uniform(low=-1.5, high=1.5, size=None)
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

        Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

        location = 0

        Spec = Dataset_Spec[location:]
        if Spec.shape[0]<num_frames:
            Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
        elif Spec.shape[0]>=num_frames:
            Spec = Spec[:num_frames]

        Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))
        Classes.append(Class)

Spec_Matrix_All = Spec_Matrix_All[1:]

np.save('data/interim/Dataset_Beatbox', Spec_Matrix_All)
np.save('data/interim/Classes_Beatbox', np.array(Classes))




# Create VIPS Dataset Reference

print('VIPS Dataset Reference')

path_audio = 'data/external/VIPS_Dataset_KSH/drum_sounds'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
Classes = []

for i in range(len(list_wav)):

    print(str(i) + ' of ' + str(len(list_wav)))

    audio, fs = librosa.load(list_wav[i], sr=44100)
    if len(audio)<frame_size:
        continue
    audio = audio/np.max(abs(audio))

    if i<2:
        Class = 'hhc'
    elif i>=2 and i<6:
        Class = 'hho'
    elif i>=6 and i<12:
        Class = 'kd'
    else:
        Class = 'sd'

    Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

    location = 0

    Spec = Dataset_Spec[location:]
    if Spec.shape[0]<num_frames:
        Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
    elif Spec.shape[0]>=num_frames:
        Spec = Spec[:num_frames]

    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))
    Classes.append(Class)

Spec_Matrix_All = Spec_Matrix_All[1:]

np.save('data/interim/Dataset_VIPS_Ref', Spec_Matrix_All)
np.save('data/interim/Classes_VIPS_Ref', np.array(Classes))





# Create VIPS Dataset Reference

print('VIPS Dataset Reference')

path_audio = 'data/external/VIPS_Dataset_KSH/imitations'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

Spec_Matrix_All = np.zeros((1,num_spec,num_frames))

for i in range(len(list_wav)):

    print(str(i) + ' of ' + str(len(list_wav)))

    audio, fs = librosa.load(list_wav[i], sr=44100)
    if len(audio)<frame_size:
        continue
    audio = audio/np.max(abs(audio))

    Dataset_Spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(2048))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

    location = 0

    Spec = Dataset_Spec[location:]
    if Spec.shape[0]<num_frames:
        Spec = np.concatenate((Spec,np.zeros((num_frames-Spec.shape[0],num_spec))))
    elif Spec.shape[0]>=num_frames:
        Spec = Spec[:num_frames]

    Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec.T,axis=0)))

Spec_Matrix_All = Spec_Matrix_All[1:]

np.save('data/interim/Dataset_VIPS_Imi', Spec_Matrix_All)
np.save('data/interim/Classes_VIPS_Imi', np.array(Classes*14))
