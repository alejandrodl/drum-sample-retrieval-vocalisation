# -*- coding: utf-8 -*-

''' Class for audio samples. This basically generates and stores a barkgram for 
    a given sound, with some associated metadata (libray, type (stim/imit/sample) etc)
'''

import sys; sys.dont_write_bytecode = True # stops pyc building

import numpy as np
import os 
import pickle 
import soundfile as sf

import barkgram as bark
import helper


class Sample(object):

    def __init__(self, input_audio, n_fft, hop_size, n_bands, bark_basis, bark_freqs, ear_basis, numpy_input=False):

        if numpy_input:

            data = input_audio.copy()
            fs = 44100

        else:

            # Set sample metadata 
            self.input_audio = input_audio
            split = self.input_audio.split('/')
            # self.type = split[2]
            # self.library = split[3]
            # self.filename = split[-1]

            # Load audio data
            data, fs = sf.read(input_audio)
            if fs != 44100:
                print('FS ERROR !!!')

        # If it's stereo then sum and scale 
        if len(data.shape) == 1:
            data = data
        elif len(data.shape) == 2: 
            data = (data[:,0]+data[:,1])*0.5
        else:
            print('ERRROR IN WAV DATA')
            data = np.zeros(99) # backup case - can filter these out later


        data = self.__trim_and_pad(data, fs, n_fft, hop_size, n_bands)
        self.__build_barkgram(data, fs, n_fft, hop_size, n_bands, bark_basis, bark_freqs, ear_basis)
        



    def __trim_and_pad(self, data, fs, n_fft, hop_size, n_bands):

        # Force target length (128 frames after stft)
        target_length = (n_bands*hop_size)-(hop_size*5)
        if len(data) > target_length:
            data = data[:target_length]

        # Apply tiny fade ins and outs if ness. (to prevent discontinuities) 
        if data[0] != 0 or data[-1] != 0:
            fade_length = fs/1000
            fade_window = np.hanning(fade_length*2)
            if data[0] != 0:
                fade = fade_window[:fade_length]
                data[:fade_length] *= fade
            if data[-1] != 0:
                fade = fade_window[fade_length:] 
                data[-fade_length:] *= fade 
        return data




    def __build_barkgram(self, data, fs, n_fft, hop_size, n_bands, bark_basis, bark_freqs, ear_basis):

        # Get barkgram 
        bg = bark.barkgram(data, fs, n_fft, hop_size, bark_basis)

        # Power to db - add eps to 0s to ensure no log(0) attempted
        smallest_val = np.nextafter(0, 1)
        bg[bg == 0.0] = smallest_val

        # Quick check that values are in range for logging
        if np.max(bg) > 1.0 or np.min(bg) <= 0.0: 
            print('Range WARNING ', self.input_audio, np.min(bg), np.max(bg)) 
            # self.plot_stft_spectrogram()

        log_bg = 20*np.log10(bg) 

        # Apply ear scaling 
        ear_bg = bark.ear_model_scaling(log_bg, ear_basis) 

        # Clip dynamic range and save
        clip_bg = np.clip(ear_bg, -144, 0)

        # Finally, 0 pad to 128 frames if needed
        if clip_bg.shape[1] < 128:
            padding = np.zeros((n_bands,n_bands-clip_bg.shape[1]))
            padding.fill(-144)
            clip_bg = np.concatenate((clip_bg, padding), axis=1)

        self.bgram = clip_bg

        # Shape check 
        if clip_bg.shape != (128,128):    
            print('Shape WARNING ', self.input_audio, self.bgram.shape)

        # helper.plot_bgram(self.bgram)  # plot it (optional)
        


