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




# Finding and excluding unreliable listeners

Listener = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=0)
Imitator = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=1)
Sound = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=2)
Imitation = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=3)
Rating = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=4)
Duplicate = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=5)

Matrix_Listening = np.zeros((len(Listener),6))
Matrix_Listening[:,0] = Listener
Matrix_Listening[:,1] = Imitator
Matrix_Listening[:,2] = Sound
Matrix_Listening[:,3] = Imitation
Matrix_Listening[:,4] = Rating
Matrix_Listening[:,5] = Duplicate

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






for n in range(Listeners_Duplicates_Scores.shape[0]):
    print(Listeners_Duplicates_Scores[n])






from scipy.stats import spearmanr

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






# Delete uncorrelated

#listeners_delete = np.where(Spearman_Rho<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho))[0].tolist()
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

Matrix_Listening = np.zeros((len(Listener),6))
Matrix_Listening[:,0] = Listener
Matrix_Listening[:,1] = Imitator
Matrix_Listening[:,2] = Sound
Matrix_Listening[:,3] = Imitation
Matrix_Listening[:,4] = Rating
Matrix_Listening[:,5] = Duplicate






print(Listener.shape)







idx_vgg = np.arange(1)+8
idx_tli = np.arange(1)+1+8
idx_adib = np.arange(1)+1+1+8
idx_tim = np.arange(10)+1+1+1+8
idx_vae = np.arange(10)+10+1+1+1+8
idx_cvae1 = np.arange(10)+10+10+1+1+1+8
idx_cvae2 = np.arange(10)+10+10+10+1+1+1+8
idx_cvae3 = np.arange(10)+10+10+10+10+1+1+1+8






string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag,vgg,tli,adib'
for n in range(10):
    if n<=9:
        string_head += ',tim_0' + str(n)
    else:
        string_head += ',tim_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae1_0' + str(n)
    else:
        string_head += ',cvae1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae2_0' + str(n)
    else:
        string_head += ',cvae2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae3_0' + str(n)
    else:
        string_head += ',cvae3_' + str(n)
        
header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

print(header_list)
print(len(header_list))






# Make data for LMER analysis

f = open('LMER_Dataset.csv','w')

string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag,vgg,tli,adib'
for n in range(10):
    if n<=9:
        string_head += ',tim_0' + str(n)
    else:
        string_head += ',tim_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae1_0' + str(n)
    else:
        string_head += ',cvae1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae2_0' + str(n)
    else:
        string_head += ',cvae2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae3_0' + str(n)
    else:
        string_head += ',cvae3_' + str(n)
f.write(string_head)
f.write('\n')

names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
for n in range(len(names_vips)):
    names_vips[n] = names_vips[n]
features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

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

# Create Matrix All

Matrix_All = np.zeros((len(Listener),522))
Matrix_All[:,0] = np.arange(len(Listener))
Matrix_All[:,1] = Trial
Matrix_All[:,2] = Listener
Matrix_All[:,3] = Imitator
Matrix_All[:,4] = Sound
Matrix_All[:,5] = Imitation
Matrix_All[:,6] = Rating
Matrix_All[:,7] = Duplicate





idx_vgg = np.arange(1)+8
idx_tli = np.arange(1)+1+8
idx_adib = np.arange(1)+1+1+8
idx_tim = np.arange(10)+1+1+1+8
idx_vae = np.arange(10)+10+1+1+1+8
idx_cvae1 = np.arange(10)+10+10+1+1+1+8
idx_cvae2 = np.arange(10)+10+10+10+1+1+1+8
idx_cvae3 = np.arange(10)+10+10+10+10+1+1+1+8


# VGG
    
c = 0

for it in idx_vgg:

    features_ref = features_vgg_ref[c]
    features_imi = features_vgg_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# TL-Iminet
    
c = 0

for it in idx_tli:

    features_ref = features_tli_ref[c]
    features_imi = features_tli_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# Adib
    
c = 0

for it in idx_adib:

    features_ref = features_adib_ref[c]
    features_imi = features_adib_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# AE

c = 0

for it in idx_ae:

    features_ref = features_ae_ref[c]
    features_imi = features_ae_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# VAE

c = 0

for it in idx_vae:

    features_ref = features_vae_ref[c]
    features_imi = features_vae_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_1

c = 0

for it in idx_cae_1:

    features_ref = features_cae_1_ref[c]
    features_imi = features_cae_1_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_1

c = 0

for it in idx_cvae_1:

    features_ref = features_cvae_1_ref[c]
    features_imi = features_cvae_1_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_2

c = 0

for it in idx_cae_2:

    features_ref = features_cae_2_ref[c]
    features_imi = features_cae_2_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_2

c = 0

for it in idx_cvae_2:

    features_ref = features_cvae_2_ref[c]
    features_imi = features_cvae_2_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_3

c = 0

for it in idx_cae_3:

    features_ref = features_cae_3_ref[c]
    features_imi = features_cae_3_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_3

c = 0

for it in idx_cvae_3:

    features_ref = features_cvae_3_ref[c]
    features_imi = features_cvae_3_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# Best

c = 0

for it in idx_best:

    features_ref = features_learnt_best_ref[c]
    features_imi = features_learnt_best_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# Write distances in CSV file
    
for i in range(Matrix_All.shape[0]):
    string = ''
    for j in range(Matrix_All.shape[1]):
        if j!=0:
            string += str(Matrix_All[i,j])+','
        else:
            string += str(int(Matrix_All[i,j]))+','
    f.write(string)
    f.write('\n')
f.close()






string_head = ',vgg,tli,adib'
for n in range(10):
    if n<=9:
        string_head += ',tim_0' + str(n)
    else:
        string_head += ',tim_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae1_0' + str(n)
    else:
        string_head += ',cvae1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae2_0' + str(n)
    else:
        string_head += ',cvae2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',cvae3_0' + str(n)
    else:
        string_head += ',cvae3_' + str(n)
string_head = string_head[1:]
        
header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

print(header_list)
print(len(header_list))

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





from scipy.stats import linregress

indices_sounds = []
for i in range(18):
    idxs = []
    for j in range(len(Sound)):
        if Sound[j]==i:
            idxs.append(j)
    indices_sounds.append(idxs)

Slopes = np.zeros((len(header_list),18))
CIs_95 = np.zeros((len(header_list),18))
CIs_99 = np.zeros((len(header_list),18))
Accuracies = np.zeros(len(header_list))

for i in range(len(header_list)):
    
    name = header_list[i]
    
    ci_ubs = np.zeros(18)
    
    for j in range(18):
        
        idxs = indices_sounds[j]
        
        x = Matrix_All[:,8+i]
        y = Matrix_All[:,6]
        
        x = np.array(x[idxs])
        y = np.array(y[idxs])
        
        Slopes[i,j], intercept, r, p, std_err = linregress(x, y)
        
        CIs_95[i,j] = 1.96*std_err
        CIs_99[i,j] = 2.58*std_err
        
        ci_ubs[j] = Slopes[i,j] + CIs_95[i,j]
        
    Accuracies[i] = 100*(len(ci_ubs[ci_ubs<0])/18)
    
    print(name + ' -> ' + str(Accuracies[i]))






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






np.mean(Accuracies[idx_mantel-8])






np.mean(Accuracies[idx_RnF-8])






np.mean(Accuracies[idx_RnV-8])






np.mean(Accuracies[idx_lda_1-8])






np.mean(Accuracies[idx_lda_2-8])






np.mean(Accuracies[idx_lda_3-8])






np.mean(Accuracies[idx_ae-8])






np.mean(Accuracies[idx_vae-8])






np.mean(Accuracies[idx_cae_1-8])






np.mean(Accuracies[idx_cvae_1-8])






np.mean(Accuracies[idx_cae_2-8])






np.mean(Accuracies[idx_cvae_2-8])






np.mean(Accuracies[idx_cae_3-8])






np.mean(Accuracies[idx_cvae_3-8])






np.mean(Accuracies[idx_best-8])






idx = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]






mantel = 52173





int(np.round(mantel))-51268





lda_1 = 52255





int(np.round(lda_1))-51268





lda_2 = 52249





int(np.round(lda_2))-51268





lda_3 = 52258





int(np.round(lda_3))-51268





learnt_ae = (51909+51868+51889+51899+51813)/5





int(np.round(learnt_ae))-51268





learnt_cae4 = (52098+51920+52124+51808+52215)/5





int(np.round(learnt_cae4))-51268





learnt_cae5 = (52145+52249+52171+52253+52025)/5





int(np.round(learnt_cae5))-51268





learnt_cae6 = (52034+51983+52188+52098+52072)/5





int(np.round(learnt_cae6))-51268





learnt_cvae4 = (51951+51827+51959+52005+51826)/5





int(np.round(learnt_cvae4))-51268





learnt_cvae5 = (51823+51763+51685+51653+52158)/5





int(np.round(learnt_cvae5))-51268





learnt_cvae6 = (51880+51974+52077+52044+52028)/5





int(np.round(learnt_cvae6))-51268





learnt_vae = (52019+51871+51837+51972+52214)/5





int(np.round(learnt_vae))-51268





learnt_best = (51564+52215+51849+51913+51697+51981+51734+52036+51799+51922)/10





int(np.round(learnt_best))-51268





randomf = (52270+52251+52258+52258+52265+52264+52267+52257+52256+52252)/10





int(np.round(randomf))-51268





# + 3





list_mod = sorted(['best_selected_models/AE_Best_Model_0.0019419_0.5520154_0.6474937_Data.npy', 'best_selected_models/AE_Best_Model_0.0019779_0.7027591_0.6972786_Data.npy', 'best_selected_models/AE_Best_Model_0.0019910_0.5794454_0.6317227_Data.npy', 'best_selected_models/AE_Best_Model_0.0019976_0.5858140_0.6936637_Data.npy', 'best_selected_models/AE_Best_Model_0.0020247_0.6377035_0.8001074_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0018811_0.6075717_0.6704741_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019115_0.5321414_0.6264842_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019709_0.6584659_0.6400411_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019837_0.5536406_0.6716249_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019891_0.4896485_0.6037273_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0022478_0.5245438_0.6090213_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024634_0.4861702_0.8282709_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024808_0.7310907_0.7196511_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024985_0.6632007_0.8056750_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0025210_0.6663925_0.6435621_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0021448_0.5602453_0.6087275_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0021985_0.6060365_0.6051637_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022245_0.7234073_0.6430170_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022417_0.6667727_0.6500232_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022530_0.6257517_0.6278596_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0018744_0.4969617_0.7022234_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019386_0.5548277_0.7498651_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019411_0.4665029_0.7550572_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019688_0.4617105_0.6786706_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019847_0.4843924_0.5987158_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019065_0.3855581_0.7854835_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019561_0.4614138_0.6662702_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019745_0.5446729_0.6515771_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0020391_0.2171588_0.6665693_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0020689_0.3854253_0.6579586_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019690_0.6665091_0.6347086_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019750_0.6599507_0.7129225_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019964_0.6643969_0.7294808_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019968_0.6298090_0.7435186_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0020425_0.6239063_0.7450548_Data.npy', 'best_selected_models/VAE_Best_Model_0.0025153_0.4545576_0.7613924_Data_0.0025799554874342.npy', 'best_selected_models/VAE_Best_Model_0.0022614_0.3910326_0.6531923_Data_0.00280211645148033.npy', 'best_selected_models/VAE_Best_Model_0.0023300_0.5525658_0.5831520_Data_0.00221217368002391.npy', 'best_selected_models/VAE_Best_Model_0.0024387_0.5522562_0.6591301_Data_0.00231544562304288.npy', 'best_selected_models/VAE_Best_Model_0.0024442_0.3588048_0.5949293_Data_0.00258621193780704.npy', 'best_selected_models/CVAE_1_Best_Model_0.0050115_0.3732934_0.5941474_Data_0.0082562468812934.npy', 'best_selected_models/CVAE_2_Best_Model_0.0047286_0.9550069_0.6135752_Data_0.0092175056422238.npy', 'best_selected_models/CVAE_2_Best_Model_0.0048150_0.9756547_0.6479091_Data_0.0055953922929573.npy', 'best_selected_models/CVAE_2_Best_Model_0.0048283_0.9850103_0.6354139_Data_0.0061535459416285.npy', 'best_selected_models/CVAE_3_Best_Model_0.0039889_0.6136144_0.9053357_Data_0.0091942633008478.npy', 'best_selected_models/CVAE_3_Best_Model_0.0050748_0.7895106_0.6543988_Data_0.0091982551213056.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025115_0.9999800_0.6462974_Data_0.0035722514512675.npy', 'best_selected_models/CVAE_1_Best_Model_0.0046143_0.3094547_0.7463438_Data_0.00579973607931406.npy', 'best_selected_models/CVAE_1_Best_Model_0.0048901_0.3018844_0.4794498_Data_0.00483585455454886.npy', 'best_selected_models/CVAE_1_Best_Model_0.0049582_0.3985969_0.6306517_Data_0.00890625837954098.npy', 'best_selected_models/CVAE_2_Best_Model_0.0046652_0.8619187_0.5895047_Data_0.00669572886678188.npy', 'best_selected_models/CVAE_2_Best_Model_0.0049057_0.7605294_0.5402406_Data_0.00698272517204962.npy', 'best_selected_models/CVAE_3_Best_Model_0.0042877_0.6838705_0.8743827_Data_0.01006980466309196.npy', 'best_selected_models/CVAE_3_Best_Model_0.0050710_0.5141815_0.7506624_Data_0.00954141663068273.npy', 'best_selected_models/CVAE_3_Best_Model_0.0051036_0.7694160_0.8065738_Data_0.00619199938058122.npy', 'best_selected_models/CVAE_4_Best_Model_0.0023684_0.4197840_0.7251867_Data_0.00267091166333954.npy', 'best_selected_models/CVAE_4_Best_Model_0.0023821_0.5435464_0.5775240_Data_0.00249365134467395.npy', 'best_selected_models/CVAE_4_Best_Model_0.0024367_0.4398358_0.4737689_Data_0.00298475316088107.npy', 'best_selected_models/CVAE_4_Best_Model_0.0025545_0.4147270_0.5639723_Data_0.00255371628626301.npy', 'best_selected_models/CVAE_5_Best_Model_0.0024744_0.9997586_0.6942574_Data_0.00322912841044494.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025540_0.9997758_0.6719118_Data_0.00331298040038967.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025541_0.9999123_0.6756387_Data_0.00473138161653001.npy', 'best_selected_models/CVAE_6_Best_Model_0.0019480_0.5628961_0.5867912_Data_0.00196174720404012.npy', 'best_selected_models/CVAE_6_Best_Model_0.0020932_0.5284657_0.6457934_Data_0.00351248208775745.npy', 'best_selected_models/CVAE_6_Best_Model_0.0021314_0.4916062_0.6571732_Data_0.00231581938410466.npy', 'best_selected_models/CVAE_6_Best_Model_0.0021363_0.4294877_0.7062503_Data_0.00231489987502567.npy', 'best_selected_models/CVAE_1_Best_Model_0.0045506_0.3440242_0.4332481_Data_0.008466356461106687.npy', 'best_selected_models/CVAE_4_Best_Model_0.0024939_0.5997672_0.6139313_Data_0.002201041840957352.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025297_0.9999482_0.7130125_Data_0.003492049574402286.npy', 'best_selected_models/CVAE_6_Best_Model_0.0020646_0.7358988_0.7277899_Data_0.002751705141896179.npy'])





list_mod[50:55]







