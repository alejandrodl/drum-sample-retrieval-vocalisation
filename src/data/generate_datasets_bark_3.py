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




# Create AVP_Fixed_Small Aug Dataset

print('AVP_Fixed_Small Aug Dataset')

Dataset_Str = 'AVP_Fixed_Small_Aug'

path_audio = '../../data/external/AVP_Dataset/Fixed'

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

    audio, fs = sf.read(list_wav[i])
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

np.save('../../data/interim/Dataset_AVP_Fixed_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_AVP_Fixed_Bark', Classes_All)







# Create LVT_2 Aug Dataset

print('LVT_2 Aug Dataset')

Dataset_Str = 'LVT2'

path_audio = '../../data/external/LVT_Dataset/DataSet_Wav_Annotation'

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

    audio, fs = sf.read(list_wav[i])
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

np.save('../../data/interim/Dataset_LVT_2_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_LVT_2_Bark', Classes_All)







# Create LVT_3 Aug Dataset

print('LVT_3 Aug Dataset')

Dataset_Str = 'LVT3'

path_audio = '../../data/external/LVT_Dataset/DataSet_Wav_Annotation'

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

    audio, fs = sf.read(list_wav[i])
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

np.save('../../data/interim/Dataset_LVT_3_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_LVT_3_Bark', Classes_All)






# Create Zhu Aug Dataset

print('Beatbox Dataset')

Dataset_Str = 'Zhu_Aug'

path_audio = '../../data/external/Beatbox_Dataset'

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

    audio, fs = sf.read(list_wav[i])
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

        Spec = Sample(audio,n_fft=4096,hop_size=512,n_bands=128,bark_basis=weights,bark_freqs=cent_freqs,ear_basis=db_diffs,numpy_input=True).bgram

        Spec_Matrix_All = np.vstack((Spec_Matrix_All,np.expand_dims(Spec,axis=0)))
        Classes.append(Class)

Spec_Matrix_All = Spec_Matrix_All[1:]

np.save('../../data/interim/Dataset_Beatbox_Bark', Spec_Matrix_All)
np.save('../../data/interim/Classes_Beatbox_Bark', np.array(Classes))