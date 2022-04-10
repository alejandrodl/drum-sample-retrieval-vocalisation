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

num_iterations = 5
modes = ['Random','Heuristic','CAE-B-Original','CAE-B','CAE','CAE-SL','CAE-DL','CAE-SDL']

# Calculate rankings

rankings = np.zeros((len(modes),num_iterations,14,18))
rankings_raw = np.zeros((len(modes),num_iterations,14,18))

for md in range(len(modes)):
    mode = modes[md]

    for it in range(num_iterations):

        if mode=='Heuristic':
            embeddings_ref = np.load('data/processed/' + mode + '/Dataset_Ref_Heuristic.npy')
            embeddings_imi_pre = np.load('data/processed/' + mode + '/Dataset_Imi_Heuristic.npy')
        elif mode=='Random':
            np.random.seed(it)
            embeddings_ref = np.random.rand(18,32)
            np.random.seed(42+it)
            embeddings_imi_pre = np.random.rand(252,32)
        else:
            embeddings_ref = np.load('data/processed/' + mode + '/embeddings_ref_' + mode + '_' + str(it) + '.npy')
            embeddings_imi_pre = np.load('data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it) + '.npy')

        if mode=='Heuristic':
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
                    distances[i,j,k] = euclidean(embeddings_ref_sample, embeddings_imi_sample)

        # Calculate rankings
        for i in range(14):
            for j in range(18):
                rankings_raw = np.argsort(distances[i,j])
                rankings[md,it,i,j] = np.where(rankings_raw==j)[0][0]

# Calculate average precision
reciprocal_ranks = np.zeros((len(modes),num_iterations))
for md in range(len(modes)):
    for it in range(num_iterations):
        reciprocal_ranks[md,it] = np.mean(np.reciprocal(rankings[md,it]+1))
    mean = np.round(np.mean(reciprocal_ranks[md]),3)
    ci95 = np.round(np.std(reciprocal_ranks[md]*(1.96/(num_iterations**(0.5)))),3)
    print('MRR ' + modes[md] + ': ' + str(mean) + ' +- ' + str(ci95))

# Plot ranking curve
colours = ['black','purple','yellow','grey','cyan','orange','lime','red']
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