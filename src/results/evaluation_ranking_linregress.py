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
num_iterations_lr = 10
modes = ['eng_mfcc_env','adib','unsupervised','RI','KSH','RI_KSH','unsupervised_bark','RI_bark','KSH_bark','RI_KSH_bark']

# Calculate rankings

rankings = np.zeros((len(modes),num_iterations,14,18))

AP = np.zeros((len(modes),num_iterations,num_iterations_lr))
REC = np.zeros((len(modes),num_iterations,num_iterations_lr,18))

mean_ap_mode = np.zeros(len(modes))
std_ap_mode = np.zeros(len(modes))

mean_rec_mode = np.zeros((len(modes),18))
std_rec_mode = np.zeros((len(modes),18))

for md in range(len(modes)):

    mode = modes[md]

    ap = 0
    rec = np.zeros(18)

    for it in range(num_iterations):

        # Load Embeddings

        if mode=='eng_mfcc_env':
            features = np.load('data/processed/' + mode + '/Dataset_VIPS_Imi_MFCC_ENV.npy')
        else:
            features = np.load('data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it) + '.npy')

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

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

        for it_lr in range(10):

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_eng = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng = np.zeros((num_models,predicted.shape[-1]))
            _average_precision_perc_eng = np.zeros((num_models,252))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=it_lr)
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
                                                _average_precision_perc_eng[cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng[cou,t] = np.sum(f_scores)/pred.shape[0]
                            #_average_precision_perc_eng[cou] /= pred.shape[0]
                            cou += 1
            precisions_perc_eng_mean = np.mean(precisions_perc_eng, axis=0)
            recalls_perc_eng_mean = np.mean(recalls_perc_eng, axis=0)
            f_scores_perc_eng_mean = np.mean(f_scores_perc_eng, axis=0)
            _average_precision_perc_eng_mean = np.mean(_average_precision_perc_eng)
            _average_precision_perc_eng_std = np.std(_average_precision_perc_eng)

            AP[md,it,it_lr] = _average_precision_perc_eng_mean
            REC[md,it,it_lr] = recalls_perc_eng_mean
        
    mean_ap_mode[md] = np.mean(np.mean(AP[md],axis=0),axis=0)
    std_ap_mode[md] = np.std(np.mean(AP[md],axis=0),axis=0)

    mean_rec_mode[md] = np.mean(np.mean(REC[md],axis=0),axis=0)
    std_rec_mode[md] = np.mean(np.mean(REC[md],axis=0),axis=0)

    print('Average Precision ' + mode + ' ' + str(it) + ': ' + str(mean_ap_mode[md]))
    print('Recall at Rank ' + mode + ' ' + str(it) + ': ' + str(mean_rec_mode[md]))
    print('')


print('')
print(mean_ap_mode)
print('')
print(std_ap_mode)
print('')
print('')
for md in range(len(modes)):
    print(mean_rec_mode[md])
    print('')
    print(std_rec_mode[md])
    print('')

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.scatter(np.arange(18)+1,mean_rec_mode[0],marker='D',edgecolor='black',s=150,c='grey',label='MFCC + Env')
plt.scatter(np.arange(18)+1,mean_rec_mode[1],marker='D',edgecolor='black',s=150,c='black',label='Unsupervised')
plt.scatter(np.arange(18)+1,mean_rec_mode[2],marker='D',edgecolor='black',s=150,c='yellow',label='Adib')
plt.scatter(np.arange(18)+1,mean_rec_mode[3],marker='D',edgecolor='black',s=150,c='cyan',label='RI')
plt.scatter(np.arange(18)+1,mean_rec_mode[4],marker='D',edgecolor='black',s=150,c='orange',label='KSH')
plt.scatter(np.arange(18)+1,mean_rec_mode[5],marker='D',edgecolor='black',s=150,c='lime',label='RI_KSH')
plt.scatter(np.arange(18)+1,mean_rec_mode[6],marker='D',edgecolor='black',s=150,c='blue',label='RI_bark')
plt.scatter(np.arange(18)+1,mean_rec_mode[7],marker='D',edgecolor='black',s=150,c='red',label='KSH_bark')
plt.scatter(np.arange(18)+1,mean_rec_mode[8],marker='D',edgecolor='black',s=150,c='green',label='RI_KSH_bark')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()
