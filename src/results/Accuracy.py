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
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from skbio.stats.distance import mantel as mantel_test
#import mantel

import torch
from torch import nn
from torch.utils import data
import torch.utils.data as utils
from sklearn.svm import SVC
from torchsummary import summary
import copy
                                                                                
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from itertools import permutations, combinations, product

from scipy.stats import spearmanr
from scipy.stats import linregress
from scipy.spatial.distance import euclidean




### Load data

with open("data/external/VIPS_Dataset_KSH/listening_test_responses.csv",'r') as f:
    with open("data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

Listener = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=0)
Imitator = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=1)
Sound = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=2)
Imitation = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=3)
Rating = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=4)
Duplicate = np.loadtxt('data/external/VIPS_Dataset_KSH/listening_test_responses_no_heading.csv', delimiter=',', usecols=5)

Matrix_Listening = np.zeros((len(Listener),6))
Matrix_Listening[:,0] = Listener
Matrix_Listening[:,1] = Imitator
Matrix_Listening[:,2] = Sound
Matrix_Listening[:,3] = Imitation
Matrix_Listening[:,4] = Rating
Matrix_Listening[:,5] = Duplicate


### Find and exclude unreliable listeners

Listener_Counter = np.zeros(int(np.max(Listener)+1))
Listeners_Duplicates_Scores = np.zeros((int(np.max(Listener)+1),2,12))
for n in range(len(Listener)):
    if Duplicate[n]==1:
        Listeners_Duplicates_Scores[int(Listener[n]),0,int(Listener_Counter[int(Listener[n])])] = Rating[n]
        for i in range(len(Listener)):
            if i!=n and (Matrix_Listening[n,:4]==Matrix_Listening[i,:4]).all():
                Listeners_Duplicates_Scores[int(Listener[i]),1,int(Listener_Counter[int(Listener[i])])] = Rating[i]
                Listener_Counter[int(Listener[n])] += 1
                if Listener[n]!=Listener[i]:
                    print('Different Listener')

Spearman_Rho = np.zeros((Listeners_Duplicates_Scores.shape[0],2))
Spearman_Pval = np.zeros((Listeners_Duplicates_Scores.shape[0],2))

for n in range(Listeners_Duplicates_Scores.shape[0]):
    ratings_1_1 = Listeners_Duplicates_Scores[n,0,:6]
    ratings_1_2 = Listeners_Duplicates_Scores[n,1,:6]
    
    ratings_2_1 = Listeners_Duplicates_Scores[n,0,6:]
    ratings_2_2 = Listeners_Duplicates_Scores[n,1,6:]
    
    rho_1, pval_1 = spearmanr(ratings_1_1,ratings_1_2)
    rho_2, pval_2 = spearmanr(ratings_2_1,ratings_2_2)
    
    Spearman_Rho[n,0] = rho_1
    Spearman_Rho[n,1] = rho_2
    
    Spearman_Pval[n,0] = pval_1
    Spearman_Pval[n,1] = pval_2

listeners_delete_1 = np.where(Spearman_Rho[:,0]<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho[:,0]))[0].tolist()
listeners_delete_2 = np.where(Spearman_Rho[:,1]<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho[:,1]))[0].tolist()
listeners_delete = sorted(list(set(listeners_delete_1)&set(listeners_delete_2)))

indices_delete = []
for n in range(len(Listener)):
    if Listener[n] in listeners_delete:
        indices_delete.append(n)
    
Listener = np.delete(Listener, indices_delete)
Imitator = np.delete(Imitator, indices_delete)
Sound = np.delete(Sound, indices_delete)
Imitation = np.delete(Imitation, indices_delete)
Rating = np.delete(Rating, indices_delete)
Duplicate = np.delete(Duplicate, indices_delete)

### Make data for LMER analysis

string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag,Heuristic,CAEB,CAE,CAESL,CAEDL,CAESDL'

header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

# Delete toms and cymbals

list_valid = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
indices_delete = []
for n in range(len(Listener)):
    if Sound[n] not in list_valid or Imitation[n] not in list_valid:
        indices_delete.append(n)
    
Listener = np.delete(Listener, indices_delete)
Imitator = np.delete(Imitator, indices_delete)
Sound = np.delete(Sound, indices_delete)
Imitation = np.delete(Imitation, indices_delete)
Rating = np.delete(Rating, indices_delete)
Duplicate = np.delete(Duplicate, indices_delete)

Sound = Sound - 6
Imitation = Imitation - 6

# Create Trial Vector

Trial = np.zeros(len(Listener))
Listener_Counter = np.zeros(int(np.max(Listener)+1))
for n in range(len(Trial)//6):
    array = Matrix_Listening[int(6*n):int(6*(n+1)),:3]
    if (array==array[0]).all():
        Trial[int(6*n):int(6*(n+1))] = Listener_Counter[int(array[0,0])]
        Listener_Counter[int(array[0,0])] += 1
    else:
        print('Not following the rule')

# Create matrix

Matrix_All = np.zeros((len(Listener),len(header_list)))
Matrix_All[:,0] = (np.arange(len(Listener))+1).astype(str)
Matrix_All[:,1] = Trial
Matrix_All[:,2] = Listener
Matrix_All[:,3] = Imitator
Matrix_All[:,4] = Sound
Matrix_All[:,5] = Imitation
Matrix_All[:,6] = Rating
Matrix_All[:,7] = Duplicate

# Compute distances

latent_dim = 32
num_iterations = 5

Accuracies_95 = np.zeros((len(header_list[7:]),num_iterations))
Accuracies_99 = np.zeros((len(header_list[7:]),num_iterations))

for it in range(num_iterations):

    f = open('results/LMER_Dataset_' + str(it) + '.csv','w')

    string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag,Heuristic,CAEB,CAE,CAESL,CAEDL,CAESDL'
    modes = ['Heuristic','CAE-B','CAE','CAE-SL','CAE-DL','CAE-SDL']
    f.write(string_head)
    f.write('\n')

    for md in range(len(modes)):

        mode = modes[md]

        # Load Embeddings
        if mode=='Heuristic':
            embeddings_ref = np.load('data/processed/' + mode + '/Dataset_VIPS_Ref_Heuristic.npy')
            embeddings_imi_pre = np.load('data/processed/' + mode + '/Dataset_VIPS_Imi_Heuristic.npy')
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

        for n in range(len(Listener)):

            embeddings_1 = embeddings_ref[int(Sound[n])]
            embeddings_2 = embeddings_imi[int(Imitator[n]),int(Imitation[n])]

            Matrix_All[n,8+md] = euclidean(embeddings_1, embeddings_2)

        Matrix_All[:,8+md] = (Matrix_All[:,8+md]-np.min(Matrix_All[:,8+md]))/(np.max(Matrix_All[:,8+md])-np.min(Matrix_All[:,8+md])+1e-16)
        
    # Save to CSV file
    for i in range(Matrix_All.shape[0]):
        string = ''
        for j in range(Matrix_All.shape[1]):
            if j!=0:
                if j!=Matrix_All.shape[1]-1:
                    string += str(Matrix_All[i,j])+','
                else:
                    string += str(Matrix_All[i,j])
            else:
                if j!=Matrix_All.shape[1]-1:
                    string += str(int(Matrix_All[i,j]))+','
                else:
                    string += str(int(Matrix_All[i,j]))
        f.write(string)
        if i!=Matrix_All.shape[0]-1:
            f.write('\n')
    f.close()




    ### Calculate accuracy

    string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag,Heuristic,CAEB,CAE,CAESL,CAEDL,CAESDL'
    string_head = string_head[1:]
            
    header_list = []
    c = 0
    for n in range(len(string_head)):
        if string_head[n]==',':
            header_list.append(string_head[c:n])
            c = n+1
    header_list.append(string_head[c:])

    string_r = ''
    for n in range(len(header_list)):
        string_r += '"' + header_list[n] + '",'
    string_r = string_r[87:-1]

    indices_sounds = []
    for i in range(18):
        idxs = []
        for j in range(len(Sound)):
            if Sound[j]==i:
                idxs.append(j)
        indices_sounds.append(idxs)

    Slopes = np.zeros((len(header_list[7:]),18))
    CIs_95 = np.zeros((len(header_list[7:]),18))

    for i in range(len(header_list[7:])):
        name = modes[i]
        ci_ubs_95 = np.zeros(18)  

        for j in range(18):
            idxs = indices_sounds[j]
            
            x = Matrix_All[:,8+i]
            y = Matrix_All[:,6]
            
            x = np.array(x[idxs])
            y = np.array(y[idxs])

            Slopes[i,j], intercept, r, p, std_err = linregress(x, y)

            CIs_95[i,j] = 1.96*std_err
            ci_ubs_95[j] = Slopes[i,j] + CIs_95[i,j]

        Accuracies_95[i,it] = 100*(len(ci_ubs_95[ci_ubs_95<0])/18)

for i in range(len(header_list[7:])):
    print('Percentage of Significant Regression Slopes ' + modes[i] + ': ' + str(np.mean(Accuracies_95[i],axis=-1)))







'''colors = ['red','blue','orange','magenta','green','purple']

CIs_95[CIs_95==np.inf] = 0
CIs_99[CIs_99==np.inf] = 0

CIs_95[CIs_95>100] = 0
CIs_99[CIs_99>100] = 0

plt.figure(figsize=(12,7))
plt.plot(np.arange(20), np.zeros(20), 'k--')
for n in range(6):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='red', capsize=8)
for n in range(6,12):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='blue', capsize=8)
for n in range(12,18):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='green', capsize=8)
plt.xticks(np.arange(18)+1)
plt.title('Regression Slopes for CVAE3 Final Model', fontsize=18)
plt.xlabel('Imitated Drum Sound', fontsize=15)
plt.ylabel('Fitted Slope', fontsize=15)
plt.xlim([0,19])'''



'''learnt_best = (51564+52215+51849+51913+51697+51981+51734+52036+51799+51922)/10
int(np.round(learnt_best))-51268

randomf = (52270+52251+52258+52258+52265+52264+52267+52257+52256+52252)/10
int(np.round(randomf))-51268'''

