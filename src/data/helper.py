# -*- coding: utf-8 -*-

''' This contains some helper functions...

    Namely for splitting the dataset for training/validation and plotting stuff
'''

import sys; sys.dont_write_bytecode = True 

import numpy as np 
import random

import Sample


''' PLOTTING FUNCTIONS 
'''





''' DATASET SPLITTING/SCALING FUNCTIONS - THESE TAKE THE RESPECTIVE SUBSETS OF DATA AND SPLIT THEM 70/30. 
'''


def scale_dataset(sounds):
    ''' Scales the entire set of sounds by max/min of all bgrams (so between 0-1). 
        NOTE: if sounds is large, this is memory intensive!
    '''
    # Scale 0-1 and expand (and add extra dim)
    smax = np.max([s.bgram for s in sounds])                 
    smin = np.min([s.bgram for s in sounds])
    for i in xrange(len(sounds)):
        spec = (sounds[i].bgram - smin) / (smax - smin)
        sounds[i].bgram = np.expand_dims(spec, axis=2) 
    return sounds


def split_samples(samples):
    ''' This splits the samples from all the sample libraries (anything not from the imitation studies).
        It creates a 70/30 split of the sounds within each library. NOTE: could split across libraries 
    '''
    train = []
    valid = []
    libs = np.unique(np.array([s.library for s in samples]))
    for lib in libs:
        lib_samples = [s for s in samples if s.library == lib]  # get the sounds for this library 
        random.shuffle(lib_samples)                             # shuffle them before splitting
        split_index = int(len(lib_samples)*0.7)
        train += lib_samples[:split_index]
        valid += lib_samples[split_index:]
    return train, valid 


def split_drums(drums):
    ''' This does a 70/30 split of imitated sounds (stims), where the stimuli and respective imitations
        are grouped together. NOTE: could select 4 from 6 of each class, ensuring equal distribution 
        across classes 
    '''
    idxs = np.array(xrange(30))
    random.shuffle(idxs)
    train_idxs = idxs[:21]
    valid_idxs = idxs[21:]
    train = [d for d in drums if int(d.filename[:-4]) in train_idxs]
    valid = [d for d in drums if int(d.filename[:-4]) in valid_idxs]
    return train, valid


def split_synths(synths):
    ''' As above, does a 70/30 split of imitated sounds (stims). However could also split within groups 
        ((indie, combi features) 
    '''
    idxs = np.array(xrange(73))
    random.shuffle(idxs)
    train_idxs = idxs[:51]
    valid_idxs = idxs[51:]
    train = [s for s in synths if int(s.filename[:-4])-1 in train_idxs] # note - we have to add 1 as the filenames are indexed from 1!
    valid = [s for s in synths if int(s.filename[:-4])-1 in valid_idxs]
    return train, valid


def split_vocalsketch(vs_sounds):
    ''' Here we just take a 168/72 split of shuffled list, but could split each sub group equally (4 subgroups)
    ''' 
    stim_names = np.unique([s.filename[:-4] for s in vs_sounds if s.type=='_STIMULI_'])
    random.shuffle(stim_names)  
    train_names = stim_names[:168]
    valid_names = stim_names[168:]
    train = [v for v in vs_sounds if v.filename[:-4].split(' - ')[0] in train_names]
    valid = [v for v in vs_sounds if v.filename[:-4].split(' - ')[0] in valid_names]
    return train, valid






''' OTHER STUFF
'''

def print_durations(audiodir):
    ''' Takes a dir path for audio files, and prints the average
        duration of all files in the directory. Assumes 44.1khz sample rate.
    '''
    lens = []
    for dirpath, dirnames, filenames in os.walk(audiodir):
        for filename in filenames:
            if filename[-3:] == 'wav' or filename[-3:] == 'Wav' or filename[-3:] == 'WAV':
                data, fs = sf.read(os.path.join(dirpath, filename))
                lens.append(len(data)/44100.0)
    print(np.mean(lens), np.median(lens))

