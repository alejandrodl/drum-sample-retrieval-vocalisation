import os
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from skbio.stats.distance import mantel



# Plot user differences

p_values = np.load('results/p_values.npy')
p_values_mean = np.mean(p_values,axis=-1)

feature_list_heuristic = ['Duration', 'DerAM', 'mLoud', 'sLoud', 'mPitch', 'sPitch', 'mSCent', 'sSCent',
                       'mMFCC_01', 'mMFCC_02', 'mMFCC_03', 'mMFCC_04', 'mMFCC_05', 'mMFCC_06', 'mMFCC_07', 'mMFCC_08', 'mMFCC_09', 'mMFCC_10', 'mMFCC_11', 'mMFCC_12',
                       'dMFCC_01', 'dMFCC_02', 'dMFCC_03', 'dMFCC_04', 'dMFCC_05', 'dMFCC_06', 'dMFCC_07', 'dMFCC_08', 'dMFCC_09', 'dMFCC_10', 'dMFCC_11', 'dMFCC_12']
feature_list_caesdl = ['Feat_01', 'Feat_02', 'Feat_03', 'Feat_04', 'Feat_05', 'Feat_06', 'Feat_07', 'Feat_08', 'Feat_09', 'Feat_10', 'Feat_11', 'Feat_12', 'Feat_13', 'Feat_14', 'Feat_15', 'Feat_16',
                       'Feat_17', 'Feat_18', 'Feat_19', 'Feat_20', 'Feat_21', 'Feat_22', 'Feat_23', 'Feat_24', 'Feat_25', 'Feat_26', 'Feat_27', 'Feat_28', 'Feat_29', 'Feat_30', 'Feat_31', 'Feat_32']
part_list = ['P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06', 'P_07', 'P_08', 'P_09', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14']
part_list_red = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
part_list_none = ['', '', '', '', '', '', '', '', '', '', '', '', '', '']


f,(ax1,ax2) = plt.subplots(2,1)
cbar_ax = f.add_axes([.91, .3, .03, .4])
g1 = sns.heatmap(p_values_mean[0,0],cmap="coolwarm",cbar_ax=cbar_ax,ax=ax1) #cbar_ax=cbar_ax
g1.set_title('Mean p-values for heuristic and CAE-SDL sets')
g1.set_xticks(np.arange(32)+0.5)
g1.set_xticklabels(feature_list_heuristic)
#g1.set_xticklabels(g1.get_xticklabels(),rotation=30)
g1.set(yticks=[])
g1.set_ylabel('Imitators')
g1.set_xlabel('')
g2 = sns.heatmap(p_values_mean[-1,2],cmap="coolwarm",cbar_ax=cbar_ax,ax=ax2) #cbar_ax=cbar_ax
g2.set_xticks(np.arange(32)+0.5)
g2.set_xticklabels(feature_list_caesdl)
#g2.set_xticklabels(g2.get_xticklabels(),rotation=30)
g2.set(yticks=[])
g2.set_ylabel('Imitators')
g2.set_xlabel('')
f.tight_layout(rect=[0, 0, .9, 1])
f.show()

f.savefig('results/mantel_figure.pdf', format='pdf')