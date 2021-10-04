#!/usr/bin/env python
# coding: utf-8


import os
import csv
import pdb
import librosa
import ml_metrics
import numpy as np
import scipy as sp
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from scipy.spatial.distance import euclidean
from itertools import permutations, combinations, product



# Parameters

num_iterations = 1
modes = ['eng_mfcc_env','adib','unsupervised','RI','KSH','RI_KSH','unsupervised_bark','RI_bark','KSH_bark','RI_KSH_bark']

# Calculate rankings

rankings = np.zeros((len(modes),num_iterations,14,18))
rankings_raw = np.zeros((len(modes),num_iterations,14,18))

for md in range(len(modes)):

    mode = modes[md]

    for it in range(num_iterations):

        # Load Embeddings
        if mode=='eng_mfcc_env':
            embeddings_ref = np.load('data/processed/' + mode + '/Dataset_VIPS_Ref_MFCC_ENV.npy')
            embeddings_imi_pre = np.load('data/processed/' + mode + '/Dataset_VIPS_Imi_MFCC_ENV.npy')
        else:
            embeddings_ref = np.load('data/processed/' + mode + '/embeddings_ref_' + mode + '_' + str(it) + '.npy')
            embeddings_imi_pre = np.load('data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it) + '.npy')

        if mode=='eng_mfcc_env':
            embeddings_all = np.vstack((embeddings_ref,embeddings_imi_pre))
            for n in range(embeddings_ref.shape[1]):
                mean = np.mean(embeddings_all[:,n])
                std = np.std(embeddings_all[:,n])
                embeddings_ref[:,n] = (embeddings_ref[:,n]-mean)/(std+1e-16)
                embeddings_imi_pre[:,n] = (embeddings_imi_pre[:,n]-mean)/(std+1e-16)

        embeddings_imi = []
        for n in range(13):
            embeddings_imi.append(embeddings_imi_pre[n*18:(n+1)*18])
        embeddings_imi.append(embeddings_imi_pre[(n+1)*18:])
        embeddings_imi = np.array(embeddings_imi)

        # Calculate distances

        distances = np.zeros((14,18,18))

        for i in range(14):

            for j in range(18):

                for k in range(18):

                    embeddings_ref_sample = embeddings_ref[j]
                    embeddings_imi_sample = embeddings_imi[i,k]

                    #embeddings_ref_sample = np.linalg.norm(embeddings_ref_sample)
                    #embeddings_imi_sample = np.linalg.norm(embeddings_imi_sample)

                    distances[i,j,k] = euclidean(embeddings_ref_sample, embeddings_imi_sample)

        # Calculate rankings

        for i in range(14):

            for j in range(18):

                rankings_raw = np.argsort(distances[i,j])
                rankings[md,it,i,j] = np.where(rankings_raw==j)[0][0]

# Calculate average precision

average_precisions = np.zeros(len(modes))
for md in range(len(modes)):
    average_precisions[md] = np.mean(np.reciprocal(rankings[md]+1))
    print('Average Precision ' + modes[md] + ': ' + str(average_precisions[md]))

# Plot ranking curve

colours = ['purple','yellow','grey','cyan','orange','lime','black','blue','red','green']

plt.figure()

for md in range(len(modes)):

    mode = modes[md]

    rank_curve = np.zeros(18)

    for it in range(num_iterations):

        accumulator = 0

        for rank in range(18):

            count_rank = np.count_nonzero(rankings[md,it]==rank)
            rank_curve[rank] += (count_rank+accumulator)/rankings[md,it].size

            accumulator += count_rank
            
    plt.scatter(np.arange(18)+1,rank_curve/num_iterations,marker='D',edgecolor='black',s=150,c=colours[md],label=mode)

plt.legend()
plt.show()





