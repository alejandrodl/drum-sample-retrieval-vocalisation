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




# RFI Individual

ap_rf_1 = 0
rec_rf_1 = np.zeros(18)

features_RFI_1_ref = np.zeros((10,18,16))
features_RFI_1_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../../data/processed/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../../data/processed/features_vips.npy')

    final_names = np.load('../../data/processed/final_names_RFI_' + str(md) + '_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_RFI_1_ref[it] = final_features_vips[:18]
    features_RFI_1_imi[it] = final_features_vips[18:]

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
    precisions_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_RFI = np.zeros((num_models,252))
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
                                        _average_precision_perc_eng_RFI[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_RFI[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_RFI[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_RFI[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_RFI[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_RFI_mean = np.mean(precisions_perc_eng_RFI, axis=0)
    recalls_perc_eng_RFI_mean = np.mean(recalls_perc_eng_RFI, axis=0)
    f_scores_perc_eng_RFI_mean = np.mean(f_scores_perc_eng_RFI, axis=0)
    _average_precision_perc_eng_RFI_mean = np.mean(_average_precision_perc_eng_RFI)
    _average_precision_perc_eng_RFI_std = np.std(_average_precision_perc_eng_RFI)

    print('')
    print(_average_precision_perc_eng_RFI_mean)
    
    ap_rf_1 += _average_precision_perc_eng_RFI_mean

    print('')
    print(recalls_perc_eng_RFI_mean)
    
    rec_rf_1 += recalls_perc_eng_RFI_mean
    
ap_rf_1 = ap_rf_1/10
rec_rf_1 = rec_rf_1/10

print('Average Precision RF 1: ' + str(ap_rf_1))
print('Recall at Rank RF 1: ' + str(rec_rf_1))
print('')





# RFI Individual

ap_rf_2 = 0
rec_rf_2 = np.zeros(18)

features_RFI_2_ref = np.zeros((10,18,16))
features_RFI_2_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../../data/processed/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../../data/processed/features_vips.npy')

    final_names = np.load('../../data/processed/final_names_RFI_2_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_RFI_2_ref[it] = final_features_vips[:18]
    features_RFI_2_imi[it] = final_features_vips[18:]

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
    precisions_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_RFI = np.zeros((num_models,252))
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
                                        _average_precision_perc_eng_RFI[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_RFI[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_RFI[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_RFI[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_RFI[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_RFI_mean = np.mean(precisions_perc_eng_RFI, axis=0)
    recalls_perc_eng_RFI_mean = np.mean(recalls_perc_eng_RFI, axis=0)
    f_scores_perc_eng_RFI_mean = np.mean(f_scores_perc_eng_RFI, axis=0)
    _average_precision_perc_eng_RFI_mean = np.mean(_average_precision_perc_eng_RFI)
    _average_precision_perc_eng_RFI_std = np.std(_average_precision_perc_eng_RFI)

    print('')
    print(_average_precision_perc_eng_RFI_mean)
    
    ap_rf_2 += _average_precision_perc_eng_RFI_mean

    print('')
    print(recalls_perc_eng_RFI_mean)
    
    rec_rf_2 += recalls_perc_eng_RFI_mean
    
ap_rf_2 = ap_rf_2/10
rec_rf_2 = rec_rf_2/10

print('Average Precision RF 2: ' + str(ap_rf_2))
print('Recall at Rank RF 2: ' + str(rec_rf_2))
print('')




# RFI Individual

ap_rf_3 = 0
rec_rf_3 = np.zeros(18)

features_RFI_3_ref = np.zeros((10,18,16))
features_RFI_3_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../../data/processed/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../../data/processed/features_vips.npy')

    final_names = np.load('../../data/processed/final_names_RFI_3_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_RFI_3_ref[it] = final_features_vips[:18]
    features_RFI_3_imi[it] = final_features_vips[18:]

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
    precisions_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_RFI = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_RFI = np.zeros((num_models,252))
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
                                        _average_precision_perc_eng_RFI[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_RFI[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_RFI[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_RFI[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_RFI[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_RFI_mean = np.mean(precisions_perc_eng_RFI, axis=0)
    recalls_perc_eng_RFI_mean = np.mean(recalls_perc_eng_RFI, axis=0)
    f_scores_perc_eng_RFI_mean = np.mean(f_scores_perc_eng_RFI, axis=0)
    _average_precision_perc_eng_RFI_mean = np.mean(_average_precision_perc_eng_RFI)
    _average_precision_perc_eng_RFI_std = np.std(_average_precision_perc_eng_RFI)

    print('')
    print(_average_precision_perc_eng_RFI_mean)
    
    ap_rf_3 += _average_precision_perc_eng_RFI_mean

    print('')
    print(recalls_perc_eng_RFI_mean)
    
    rec_rf_3 += recalls_perc_eng_RFI_mean
    
ap_rf_3 = ap_rf_3/10
rec_rf_3 = rec_rf_3/10

print('Average Precision RF 3: ' + str(ap_rf_3))
print('Recall at Rank RF 3: ' + str(rec_rf_3))
print('')
