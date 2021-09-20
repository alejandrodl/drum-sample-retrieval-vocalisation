import os
import csv
import numpy as np
import soundfile as sf
from math import isclose
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.ensemble import ExtraTreesClassifier



for md in range(3):

    for it in range(10):

        Select_Features_Bool = False

        features_kick = np.load('data/processed/features_kick.npy')
        features_snare = np.load('data/processed/features_snare.npy')
        features_hihat = np.load('data/processed/features_hihat.npy')
        features_kick_imi = np.load('data/processed/features_kick_imi.npy')
        features_snare_imi = np.load('data/processed/features_snare_imi.npy')
        features_hihat_imi = np.load('data/processed/features_hihat_imi.npy')

        features = np.vstack((features_kick, features_snare, features_hihat, features_kick_imi, features_snare_imi, features_hihat_imi))
        
        if md==0:
            classes = np.concatenate((np.zeros(features_kick.shape[0]), np.zeros(features_snare.shape[0]), np.zeros(features_hihat.shape[0]), np.ones(features_kick_imi.shape[0]), np.ones(features_snare_imi.shape[0]), np.ones(features_hihat_imi.shape[0])))
        elif md==1:
            classes = np.concatenate((np.zeros(features_kick.shape[0]), np.ones(features_snare.shape[0]), 2*np.ones(features_hihat.shape[0]), np.zeros(features_kick_imi.shape[0]), np.ones(features_snare_imi.shape[0]), 2*np.ones(features_hihat_imi.shape[0])))
        elif md==2:
            classes = np.concatenate((np.zeros(features_kick.shape[0]), np.ones(features_snare.shape[0]), 2*np.ones(features_hihat.shape[0]), 3*np.ones(features_kick_imi.shape[0]), 4*np.ones(features_snare_imi.shape[0]), 5*np.ones(features_hihat_imi.shape[0])))

        names = np.load('data/processed/names_ksh.npy')

        for n in range(len(names)):
            names[n] = names[n]

        c = 0
        indices = []
        for n in range(len(names)):
            if ('barkbands' in names[n]) or ('cov' in names[n]) or ('decrease' in names[n]) or ('contrast' in names[n]) or ('erbbands' in names[n]) or ('silence_rate' in names[n]) or ('gfcc' in names[n]) or ('melbands' in names[n]):
                indices.append(n)

        indices_set = set(indices)
        names = [i for j, i in enumerate(names) if j not in indices_set]
        features = np.delete(features, indices, axis=1)

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)

        print(features.shape)

        # Forest importances

        X = features
        y = classes

        # Build a forest and compute the impurity-based feature importances
        forest = ExtraTreesClassifier(n_estimators=500)

        forest.fit(X, y)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]

        final_names = []
        for n in range(16):
            final_names.append(names[indices[n]])

        np.save('data/processed/selected_engineered/final_names_RFI_' + str(md+1) + '_' + str(it), np.array(final_names))