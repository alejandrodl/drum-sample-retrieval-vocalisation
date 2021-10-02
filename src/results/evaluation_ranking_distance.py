#!/usr/bin/env python
# coding: utf-8



import os
import csv
import pdb
import librosa
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
modes = ['unsupervised','RI','KSH','RI_KSH']

# Calculate rankings

rankings = np.zeros((len(modes),num_iterations,14,18))

for md in range(len(modes)):

    mode = modes[md]

    for it in range(num_iterations):

        # Load Embeddings

        embeddings_ref = np.load('data/processed/' + mode + '/embeddings_ref_' + mode + '_' + str(it) + '.npy')
        embeddings_imi_pre = np.load('data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it) + '.npy')
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

                    embeddings_ref_sample = np.linalg.norm(embeddings_ref_sample)
                    embeddings_imi_sample = np.linalg.norm(embeddings_imi_sample)

                    distances[i,j,k] = euclidean(embeddings_ref_sample, embeddings_imi_sample)

        # Calculate rankings

        for i in range(14):

            for j in range(18):

                rankings[md,it,i,j] = np.where(np.argsort(distances[i,j])==j)[0][0]

# Plot ranking curve

colours = ['black','blue','red','green']

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




