# -*- coding: utf-8 -*-

''' Here are the functions used to create bark grams. Note - the data is not 0 padded prior to the stft, and any samples 
    in the last frame are truncated if it's not full. This is because we use a fixed length for the cnn and the 
    files are 0 padded or truncated prior to the barkgrams being calculated. 
'''

import soundfile as sf
import numpy as np
import math 



def calc_bark_spaced_cent_freqs(n_bands):
    ''' For a given number of bands, calculates the bark spaced cutoff frequencies from 0-24.9 barks (approx fs/2 at 
        44.1k. As we use triangular overlapping windows, each cutoff is a centre frequencie for the previous band, 
        so just take the cutoff frequencies to be centre frequencies. 

        We calculate n_bands+2 centre frequencies as the first and last are only used as cutoffs in the basis matrix. 

        Equations taken from: (Traunmüller 1990): https://ccrma.stanford.edu/courses/120-fall-2003/lecture-5.html
        
        NOTE2: there is a discontinuity in the bandwidths when jumps between z >= 2 and z > 20.1. This does not 
        matter on the low end, but does on the high end (there is a slight dip in bandwidth)
    '''
    # list of bark values 
    barks = np.linspace(0,24.9,n_bands+2)
    cent_freqs = []
    # now set the rest of the cent freqs based on the calculation for bark cutoff frequencies 
    for z in barks:
        if z < 2.0: 
            z_prime = 2.0 + ((z-2.0) / 0.85)
        elif z > 20.1:
            z_prime = 20.1 + ((z-20.1) / 1.22)  
        else:
            z_prime = z
        freq = (1960.0*(z_prime+0.53)) / (26.28-z_prime)  
        cent_freqs.append(freq)
    return np.array(cent_freqs)



def bin_frequencies(fs, n_fft):
    ''' Calculates the center frequencies of the fft bins 
    '''
    return (float(fs)/n_fft) * np.arange(1+n_fft/2) 


def bark_basis(fs, n_fft, n_bands):
    ''' Creates a Filterbank matrix to combine FFT bins into Mel-frequency bins
    '''
    # Init the weight matrix  
    weights = np.zeros((n_bands, int(1 + n_fft // 2)))
    # Calculate the bark frequencies 
    bark_freqs = calc_bark_spaced_cent_freqs(n_bands)
    # Get the bin frequencies
    bin_freqs = bin_frequencies(fs, n_fft)
    # Calculate the bandwidths
    fdiff = np.diff(bark_freqs)
    # Calculate the lower and upper slopes for all bins
    ramps = np.subtract.outer(bark_freqs, bin_freqs)
    # Calculate scaling to ensure bgram is approx constant energy per channel
    enorm = np.min(fdiff) / fdiff
    # Now calculate the weights
    for i in range(n_bands):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]
        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
    return weights



def stft(data, fs, n_fft, hop_size):
    ''' STFT function. Pads start and end to centre first window and 
        ensure all samples included in stft
    '''
    data = np.insert(data,0,np.zeros(n_fft/2))                                              # start pad
    data = np.concatenate((data, np.zeros(hop_size-(len(data)%hop_size))))                  # end pad (to make integer multiple of hop)
    data = np.concatenate((data, np.zeros(n_fft-hop_size)))                                 # end pad (to ensure all samples included in spec)            
    total_frames = np.int32(np.floor(len(data)/ np.float32(hop_size)))-(n_fft/hop_size)+1    
    window = np.hanning(n_fft)                                                              # half cosine window
    spec = np.empty((total_frames, (n_fft/2)+1), dtype=np.float32)                          # space to hold the spectrogram
    for i in xrange(total_frames):                       
        current_hop = hop_size * i                          
        frame = data[current_hop:current_hop+n_fft]         # get the current frame
        windowed = frame * window                           # apply window 
        spectrum = np.fft.fft(windowed) / n_fft             # take the Fourier Transform and scale by the number of actual samples
        autopower = np.abs(spectrum * np.conj(spectrum))    # take the autopower spectrum
        spec[i, :] = autopower[:(n_fft/2)+1]                # remove neg freqs
    return spec.T



def barkgram(data, fs, n_fft, hop_size, bark_basis):
    ''' Create a bark spectrogram with triangular overlapping bands
    '''
    spec = stft(data, fs, n_fft, hop_size) # shape should be NxM, N=n_bins, M=n_frames
    barkgram = np.dot(bark_basis, spec)
    return barkgram



def ear_model_basis(freqs):
    ''' Calculates the basis function with which to multiply the frequency band magnitudes, 
        based on equation of Terhardt's ear model (http://web.media.mit.edu/~tristan/phd/dissertation/chapter3.html)
    '''    
    db_diffs = []
    for i in xrange(len(freqs)):
        f = freqs[i] * 0.001 # convert to kHz
        db_diffs.append((-3.64 * pow(f, -0.8)) + (6.5 * math.exp(-0.6 * pow(f-3.3, 2.))) - (pow(10., -3.) * pow(f, 4.)))
    return np.array(db_diffs)


def ear_model_scaling(log_bg, ear_basis):
    ''' Applies basis function to input data (expects the input to be scaled in db already

        Expects stft/bark/mel frames in NxM where N = freqs, and M = time frames.

        EXAMPLE:
        m = np.array([[1,2],[3,4],[5,6]])
        v = np.array([10,100,1000])
        scaled= (m.T+v).T
    '''
    return (log_bg.T+ear_basis).T
