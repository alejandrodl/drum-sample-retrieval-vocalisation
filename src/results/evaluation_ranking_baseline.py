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

from itertools import permutations, combinations, product

from models.networks import *




# Random Features

final_scores_perc_eng_RandomF_mean = np.zeros(18)
_recalls_randomf_1 = 0
_recalls_randomf_3 = 0
_recalls_randomf_5 = 0
_recalls_randomf_15 = 0
_average_precision_perc_eng_RandomF_mean_all = 0

features_randomf_ref = np.zeros((10,18,16))
features_randomf_imi = np.zeros((10,252,16))

for it in range(10):
    
    names_vips = np.load('../../data/processed/names_vips.npy')
    features_vips = np.load('../../data/processed/features_vips.npy')
    
    np.random.seed(it)
    indices = np.random.random(len(names_vips))
    indices = indices.argsort()[:16].tolist()
    
    final_features_vips = features_vips[:,indices]
    
    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_randomf_ref[it] = final_features_vips[:18]
    features_randomf_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_RandomF = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_RandomF[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_RandomF[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_RandomF[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_RandomF[cou,t] = np.sum(f_scores)/pred.shape[0]
                    cou += 1
    precisions_perc_eng_RandomF_mean = np.mean(precisions_perc_eng_RandomF, axis=0)
    recalls_perc_eng_RandomF_mean = np.mean(recalls_perc_eng_RandomF, axis=0)
    f_scores_perc_eng_RandomF_mean = np.mean(f_scores_perc_eng_RandomF, axis=0)
    _average_precision_perc_eng_RandomF_mean = np.mean(_average_precision_perc_eng_RandomF)
    _average_precision_perc_eng_RandomF_std = np.std(_average_precision_perc_eng_RandomF)

    print(precisions_perc_eng_RandomF_mean)
    print(recalls_perc_eng_RandomF_mean)
    print(f_scores_perc_eng_RandomF_mean)
    print('')

    print(_average_precision_perc_eng_RandomF_mean)
    print(_average_precision_perc_eng_RandomF_std)
    print(max(f_scores_perc_eng_RandomF_mean))

    print('')
    print((precisions_perc_eng_RandomF_mean[0]+precisions_perc_eng_RandomF_mean[2]+precisions_perc_eng_RandomF_mean[4])/3)
    print((recalls_perc_eng_RandomF_mean[0]+recalls_perc_eng_RandomF_mean[2]+recalls_perc_eng_RandomF_mean[4])/3)
    print((f_scores_perc_eng_RandomF_mean[0]+f_scores_perc_eng_RandomF_mean[2]+f_scores_perc_eng_RandomF_mean[4])/3)

    plt.plot(precisions_perc_eng_RandomF_mean)
    plt.plot(recalls_perc_eng_RandomF_mean)
    plt.plot(f_scores_perc_eng_RandomF_mean)

    _recalls_randomf_1 += recalls_perc_eng_RandomF_mean[0]
    _recalls_randomf_3 += recalls_perc_eng_RandomF_mean[2]
    _recalls_randomf_5 += recalls_perc_eng_RandomF_mean[4]
    _recalls_randomf_15 += (recalls_perc_eng_RandomF_mean[0]+recalls_perc_eng_RandomF_mean[2]+recalls_perc_eng_RandomF_mean[4])/3
    _average_precision_perc_eng_RandomF_mean_all += _average_precision_perc_eng_RandomF_mean
    final_scores_perc_eng_RandomF_mean += recalls_perc_eng_RandomF_mean.copy()
    
rec_randomfeat = final_scores_perc_eng_RandomF_mean/10
ap_randomfeat = _average_precision_perc_eng_RandomF_mean_all/10

print('Average Precision Random Features: ' + str(rec_randomfeat))
print('Recall at Rank Random Features: ' + str(ap_randomfeat))
print('')





# Random

final_scores_perc_eng_Random_mean = np.zeros(18)
_recalls_random_1 = 0
_recalls_random_3 = 0
_recalls_random_5 = 0
_recalls_random_15 = 0
_average_precision_perc_eng_Random_mean_all = 0

features_randomv_ref = np.zeros((10,18,16))
features_randomv_imi = np.zeros((10,252,16))

for it in range(10):

    np.random.seed(it)
    final_features_vips = np.random.random((270,16))
    
    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_randomv_ref[it] = final_features_vips[:18]
    features_randomv_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_Random = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_Random[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_Random[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_Random[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_Random[cou,t] = np.sum(f_scores)/pred.shape[0]
                    cou += 1
    precisions_perc_eng_Random_mean = np.mean(precisions_perc_eng_Random, axis=0)
    recalls_perc_eng_Random_mean = np.mean(recalls_perc_eng_Random, axis=0)
    f_scores_perc_eng_Random_mean = np.mean(f_scores_perc_eng_Random, axis=0)
    _average_precision_perc_eng_Random_mean = np.mean(_average_precision_perc_eng_Random)
    _average_precision_perc_eng_Random_std = np.std(_average_precision_perc_eng_Random)

    print(precisions_perc_eng_Random_mean)
    print(recalls_perc_eng_Random_mean)
    print(f_scores_perc_eng_Random_mean)
    print('')

    print(_average_precision_perc_eng_Random_mean)
    print(_average_precision_perc_eng_Random_std)
    print(max(f_scores_perc_eng_Random_mean))

    print('')
    print((precisions_perc_eng_Random_mean[0]+precisions_perc_eng_Random_mean[2]+precisions_perc_eng_Random_mean[4])/3)
    print((recalls_perc_eng_Random_mean[0]+recalls_perc_eng_Random_mean[2]+recalls_perc_eng_Random_mean[4])/3)
    print((f_scores_perc_eng_Random_mean[0]+f_scores_perc_eng_Random_mean[2]+f_scores_perc_eng_Random_mean[4])/3)

    plt.plot(precisions_perc_eng_Random_mean)
    plt.plot(recalls_perc_eng_Random_mean)
    plt.plot(f_scores_perc_eng_Random_mean)

    _recalls_random_1 += recalls_perc_eng_Random_mean[0]
    _recalls_random_3 += recalls_perc_eng_Random_mean[2]
    _recalls_random_5 += recalls_perc_eng_Random_mean[4]
    _recalls_random_15 += (recalls_perc_eng_Random_mean[0]+recalls_perc_eng_Random_mean[2]+recalls_perc_eng_Random_mean[4])/3
    _average_precision_perc_eng_Random_mean_all += _average_precision_perc_eng_Random_mean
    final_scores_perc_eng_Random_mean += recalls_perc_eng_Random_mean.copy()
    
rec_randomval = final_scores_perc_eng_Random_mean/10
ap_randomval = _average_precision_perc_eng_Random_mean_all/10

print('Average Precision Random Values: ' + str(rec_randomval))
print('Recall at Rank Random Values: ' + str(ap_randomval))
print('')
