import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from skbio.stats.distance import mantel



# Parameters

latent_dim = 32

num_iterations = 1
modes = ['eng_mfcc_env','adib','unsupervised','RI','KSH','RI_KSH']

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

# Perform Mantel test

num_tests = 5

mantel_scores = np.zeros((len(modes),num_iterations,14,latent_dim,num_tests))
p_values = np.zeros((len(modes),num_iterations,14,latent_dim,num_tests))

print('Performing tests...')

for md in range(len(modes)):
    mode = modes[md]
    for it in range(num_iterations):
        for i in range(14):
            for emb in range(latent_dim):
                for t in range(num_tests):
                    score, p_value, _ = mantel(distance_matrices_ref[md,it,emb],distance_matrices_imi[md,it,i,emb])
                    mantel_scores[md,it,i,emb,t] = score
                    p_values[md,it,i,emb,t] = p_value
    np.save('results/mantel_scores', mantel_scores)
    np.save('results/p_values', p_values)
    print('Done for ' + mode)

# Calculate percentage of significant Mantel scores per participant

for md in range(len(modes)):
    print('Percentage of Significant Mantel Scores ' + modes[md] + ': ' + str((p_values[md]<0.05).sum()/p_values[md].size))

'''
# Plot user differences

p_values = np.load('results/p_values.npy')
p_values_mean = np.mean(p_values,axis=-1)

plt.figure()
plt.imshow(p_values_mean[5,0])
plt.show()

for md in range(len(modes)):
    plt.figure()
    plt.imshow(p_values_mean[md,0])
    plt.show()

for md in range(len(modes)):
    matrix = p_values_mean[md,0]
    corr_agg = 0
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            corr, _ = sp.stats.pearsonr(matrix[:,i], matrix[:,j])
            corr_agg += corr
    print('Mean Pearson Correlation for ' + modes[md] + ': ' + str(corr_agg/(matrix.shape[1]**2)))
'''