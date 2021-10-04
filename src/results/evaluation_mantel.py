import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from skbio.stats.distance import mantel



# Parameters

latent_dim = 16

num_iterations = 1
modes = ['eng_mfcc_env','adib','unsupervised','RI','KSH','RI_KSH','unsupervised_bark','RI_bark','KSH_bark','RI_KSH_bark']

# Build distance matrices

distance_matrices_ref = np.zeros((len(modes),num_iterations,latent_dim,18,18))
distance_matrices_imi = np.zeros((len(modes),num_iterations,14,latent_dim,18,18))

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

        for i in range(14):

            for emb in range(latent_dim):

                for j in range(18):

                    for k in range(18):

                        distance_matrices_ref[md,it,emb,j,k] = euclidean(embeddings_ref[j,emb], embeddings_ref[k,emb])
                        distance_matrices_imi[md,it,i,emb,j,k] = euclidean(embeddings_imi[i,j,emb], embeddings_imi[i,k,emb])

# Calculate mean Mantel scores

mantel_scores = np.zeros((len(modes),num_iterations,14,latent_dim))
p_values = np.zeros((len(modes),num_iterations,14,latent_dim))

for md in range(len(modes)):
    mode = modes[md]
    for it in range(num_iterations):
        for i in range(14):
            print([mode,i])
            for emb in range(latent_dim):
                score, p_value, _ = mantel(distance_matrices_ref[md,it,emb],distance_matrices_imi[md,it,i,emb])
                mantel_scores[md,it,i,emb] = score
                p_values[md,it,i,emb] = p_value
    np.save('results/mantel_scores', mantel_scores)
    np.save('results/p_values', p_values)
