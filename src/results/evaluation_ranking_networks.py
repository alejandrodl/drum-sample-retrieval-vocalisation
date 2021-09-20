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





Dataset_Ref = np.load('data/interim/Dataset_VIPS_Ref.npy')
Dataset_Imi = np.load('data/interim/Dataset_VIPS_Imi.npy')









num_iter = 10
latent_dim = 32

tols = [1e-3,1e-4,1e-5]
reg_strs = [0.75,1.0,1.25]
solvers = ['newton-cg', 'lbfgs']
max_iters = [100, 200]

frame_size = 1024
size = 70








# Compute results

model_types = ['VAE','CVAE_1','CVAE_2','CVAE_3','CNN_1','CNN_2','CNN_3']

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_vae_imi = np.zeros(len(model_types), (len(list_models), 252, latent_dim))

AP = np.zeros(len(model_types))
REC = np.zeros((len(model_types),18))

for model_type_idx in range(len(model_types)):

    model_type = model_types[model_type_idx]

    models_mean_precisions = np.zeros((len(list_models)//num_iter,18))
    models_mean_recalls = np.zeros((len(list_models)//num_iter,18))
    models_mean_f_scores = np.zeros((len(list_models)//num_iter,18))

    models_std_precisions = np.zeros((len(list_models)//num_iter,18))
    models_std_recalls = np.zeros((len(list_models)//num_iter,18))
    models_std_f_scores = np.zeros((len(list_models)//num_iter,18))

    average_precisions = np.zeros((len(list_models)//num_iter,252))

    P_avg = np.zeros((len(list_models),18))
    R_avg = np.zeros((len(list_models),18))
    F_avg = np.zeros((len(list_models),18))

    P_std = np.zeros((len(list_models),18))
    R_std = np.zeros((len(list_models),18))
    F_std = np.zeros((len(list_models),18))

    for j in range(len(frame_sizes)):

        frame_size = frame_sizes[j]
        
        precisions_perc_lrnt = np.zeros((num_iter,18))
        recalls_perc_lrnt = np.zeros((num_iter,18))
        f_scores_perc_lrnt = np.zeros((num_iter,18))
        
        for k in range(num_iter):

            params = np.load(list_models[int((j*num_iter)+k)], allow_pickle=True)

            name = list_models[int((j*num_iter)+k)]
            model_type = name[idx_name:name.find('B')-1]

            if frame_size==512:
                num_mels = 128
            else:
                num_mels = 64

            if model_type=='VAE'
                classes = torch.zeros(18)
            elif model_type=='CVAE_1' or model_type=='CNN_1':
                classes = torch.zeros(18)
            elif model_type=='CVAE_2' or model_type=='CNN_2':
                classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
            elif model_type=='CVAE_3' or model_type=='CNN_3':
                classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))

            if model_type=='VAE':
                model = VAE_Interim(latent_dim, num_mels)
            elif model_type=='CVAE_1':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=2)
            elif model_type=='CVAE_2':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=3)
            elif model_type=='CVAE_3':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=6)
            elif model_type=='CNN_1':
                model = CNN_Interim(num_mels, num_labels=2)
            elif model_type=='CNN_2':
                model = CNN_Interim(num_mels, num_labels=3)
            elif model_type=='CNN_3':
                model = CNN_Interim(num_mels, num_labels=6)

            model.built = True
            model.load_weights(list_models[int((j*num_iter)+k)])

            if model_type=='VAE' or model_type=='CVAE_1' or model_type=='CVAE_2' or model_type=='CVAE_3':
                features = np.zeros((14,18,latent_dim))
                for n in range(14):
                    specs = tf.convert_to_tensor(Dataset_Imi[n])
                    lat, logvar = model.encode(specs, classes)
                    features[n] = lat.numpy()
            else:
                features = np.zeros((14,18,latent_dim))
                for n in range(14):
                    specs = tf.convert_to_tensor(Dataset_Imi[n])
                    feature_extractor = tf.keras.Sequential()
                    for layer in model.layers[:-3]: # go through until last layer
                        feature_extractor.add(layer)
                    feature_extractor.built = True
                    lat, logvar = feature_extractor.predict(specs)
                    features[n] = lat.numpy()

            features_flat = features.copy()
            features = np.zeros((252,latent_dim))
            for n in range(14):
                features[int(n*18):int((n+1)*18)] = features_flat[n]
        
            #features_randomf_imi[9] = features.copy() ############### RARO RARUNO... #################

            for n in range(features.shape[1]):
                features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
            features = (features-np.mean(features))/(np.std(features)+1e-16)

            features_vae_imi[int((j*num_iter)+k)] = features

            # Logistic Regression Train

            classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
            classes = np.array(classes)

            np.random.seed(0)
            np.random.shuffle(features)

            np.random.seed(0)
            np.random.shuffle(classes)

            X = features.copy()
            y = classes.copy()
            
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
                    
            cou = 0
            predicted = np.zeros((num_models,252,18))
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
                                    for w in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            models_mean_precisions[j] += precisions_perc_lrnt_mean
            models_mean_recalls[j] += recalls_perc_lrnt_mean
            models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            std_precisions[j] += precisions_perc_lrnt_std
            std_recalls[j] += recalls_perc_lrnt_std
            std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_iter)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_iter)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_iter)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_iter)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_iter)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_iter)+k] = f_scores_perc_lrnt_std

        print(model_type)
        
        mean_precisions[j] /= num_iter
        mean_recalls[j] /= num_iter
        mean_f_scores[j] /= num_iter
        
        std_precisions[j] /= num_iter
        std_recalls[j] /= num_iter
        std_f_scores[j] /= num_iter

    features_vae_ref = np.zeros((len(list_models), 18, latent_dim))

    for j in range(len(frame_sizes)):

        frame_size = frame_sizes[j]
        
        precisions_perc_lrnt = np.zeros((num_iter,18))
        recalls_perc_lrnt = np.zeros((num_iter,18))
        f_scores_perc_lrnt = np.zeros((num_iter,18))
        
        for k in range(num_iter):

            params = np.load(list_models[int((j*num_iter)+k)], allow_pickle=True)

            name = list_models[int((j*num_iter)+k)]
            model_type = name[idx_name:name.find('B')-1]

            if frame_size==512:
                num_mels = 128
            else:
                num_mels = 64

            if model_type=='VAE'
                classes = torch.ones(18)
            elif model_type=='CVAE_1' or model_type=='CNN_1':
                classes = torch.ones(18)
            elif model_type=='CVAE_2' or model_type=='CNN_2':
                classes = torch.cat((5*torch.ones(6),4*torch.ones(6),3*torch.ones(6)))
            elif model_type=='CVAE_3' or model_type=='CNN_3':
                classes = torch.cat((5*torch.ones(6),4*torch.ones(6),3*torch.ones(6)))

            if model_type=='VAE':
                model = VAE_Interim(latent_dim, num_mels)
            elif model_type=='CVAE_1':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=2)
            elif model_type=='CVAE_2':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=3)
            elif model_type=='CVAE_3':
                model = CVAE_Interim(latent_dim, num_mels, num_labels=6)
            elif model_type=='CNN_1':
                model = CNN_Interim(num_mels, num_labels=2)
            elif model_type=='CNN_2':
                model = CNN_Interim(num_mels, num_labels=3)
            elif model_type=='CNN_3':
                model = CNN_Interim(num_mels, num_labels=6)

            model.built = True
            model.load_weights(list_models[int((j*num_iter)+k)])

            if model_type=='VAE' or model_type=='CVAE_1' or model_type=='CVAE_2' or model_type=='CVAE_3':
                features = np.zeros((14,18,latent_dim))
                for n in range(1):
                    specs = tf.convert_to_tensor(Dataset_Ref[n])
                    lat, logvar = model.encode(specs, classes)
                    features[n] = lat.numpy()
            else:
                features = np.zeros((14,18,latent_dim))
                for n in range(1):
                    specs = tf.convert_to_tensor(Dataset_Ref[n])
                    feature_extractor = tf.keras.Sequential()
                    for layer in model.layers[:-3]: # go through until last layer
                        feature_extractor.add(layer)
                    feature_extractor.built = True
                    lat, logvar = feature_extractor.predict(specs)
                    features[n] = lat.numpy()

            features_flat = features.copy()
            features = np.zeros((252,latent_dim))
            for n in range(14):
                features[int(n*18):int((n+1)*18)] = features_flat[n]
        
            #features_randomf_imi[9] = features.copy() ############### RARO RARUNO... #################

            for n in range(features.shape[1]):
                features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
            features = (features-np.mean(features))/(np.std(features)+1e-16)

            features_vae_ref[int((j*num_iter)+k)] = features

    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1),axis=0))
    recalls_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1),axis=0)

    AP[model_type_idx] = np.mean(average_precisions)
    REC[model_type_idx] = recalls_perc_lrnt_mean_best.copy()







AP_VAE = AP[0]
AP_CVAE_1 = AP[1]
AP_CVAE_2 = AP[2]
AP_CVAE_3 = AP[3]
AP_CNN_1 = AP[4]
AP_CNN_2 = AP[5]
AP_CNN_3 = AP[6]

REC_VAE = REC[0]
REC_CVAE_1 = REC[1]
REC_CVAE_2 = REC[2]
REC_CVAE_3 = REC[3]
REC_CNN_1 = REC[4]
REC_CNN_2 = REC[5]
REC_CNN_3 = REC[6]







np.save('final_results_ranking/REC_Mantel', REC_Mantel)
np.save('final_results_ranking/rec_randomfeat', rec_randomfeat)
np.save('final_results_ranking/rec_randomval', rec_randomval)
np.save('final_results_ranking/rec_rf_1', rec_rf_1)
np.save('final_results_ranking/rec_rf_2', rec_rf_2)
np.save('final_results_ranking/rec_rf_3', rec_rf_3)
np.save('final_results_ranking/REC_VAE', REC_VAE)
np.save('final_results_ranking/REC_CVAE_1', REC_CVAE_1)
np.save('final_results_ranking/REC_CVAE_2', REC_CVAE_2)
np.save('final_results_ranking/REC_CVAE_3', REC_CVAE_3)
np.save('final_results_ranking/REC_CNN_1', REC_CNN_1)
np.save('final_results_ranking/REC_CNN_2', REC_CNN_2)
np.save('final_results_ranking/REC_CNN_3', REC_CNN_3)

np.save('final_results_ranking/AP_Mantel', AP_Mantel)
np.save('final_results_ranking/ap_randomfeat', ap_randomfeat)
np.save('final_results_ranking/ap_randomval', ap_randomval)
np.save('final_results_ranking/ap_rf_1', ap_rf_1)
np.save('final_results_ranking/ap_rf_2', ap_rf_2)
np.save('final_results_ranking/ap_rf_3', ap_rf_3)
np.save('final_results_ranking/AP_VAE', AP_VAE)
np.save('final_results_ranking/AP_CVAE_1', AP_CVAE_1)
np.save('final_results_ranking/AP_CVAE_2', AP_CVAE_2)
np.save('final_results_ranking/AP_CVAE_3', AP_CVAE_3)
np.save('final_results_ranking/AP_CNN_1', AP_CNN_1)
np.save('final_results_ranking/AP_CNN_2', AP_CNN_2)
np.save('final_results_ranking/AP_CNN_3', AP_CNN_3)









print([AP_Mantel,ap_randomfeat,ap_randomval,ap_rf_1,ap_rf_2,ap_rf_3,AP_AE,AP_VAE,AP_CAE_1,AP_CVAE_1,AP_CAE_2,AP_CVAE_2,AP_CAE_3,AP_CVAE_3])
print('')
print([REC_Mantel[0],rec_randomfeat[0],rec_randomval[0],rec_rf_1[0],rec_rf_2[0],rec_rf_3[0],REC_AE[0],REC_VAE[0],REC_CAE_1[0],REC_CVAE_1[0],REC_CAE_2[0],REC_CVAE_2[0],REC_CAE_3[0],REC_CVAE_3[0]])
print('')
print([REC_Mantel[2],rec_randomfeat[2],rec_randomval[2],rec_rf_1[2],rec_rf_2[2],rec_rf_3[2],REC_AE[2],REC_VAE[2],REC_CAE_1[2],REC_CVAE_1[2],REC_CAE_2[2],REC_CVAE_2[2],REC_CAE_3[2],REC_CVAE_3[2]])
print('')
print([REC_Mantel[4],rec_randomfeat[4],rec_randomval[4],rec_rf_1[4],rec_rf_2[4],rec_rf_3[4],REC_AE[4],REC_VAE[4],REC_CAE_1[4],REC_CVAE_1[4],REC_CAE_2[4],REC_CVAE_2[4],REC_CAE_3[4],REC_CVAE_3[4]])
print('')
print([np.mean(REC_Mantel[:4]),np.mean(rec_randomfeat[:4]),np.mean(rec_randomval[:4]),np.mean(rec_rf_1[:4]),np.mean(rec_rf_2[:4]),np.mean(rec_rf_3[:4]),np.mean(REC_AE[:4]),np.mean(REC_VAE[:4]),np.mean(REC_CAE_1[:4]),np.mean(REC_CVAE_1[:4]),np.mean(REC_CAE_2[:4]),np.mean(REC_CVAE_2[:4]),np.mean(REC_CAE_3[:4]),np.mean(REC_CVAE_3[:4])])






print([AP_Mantel,ap_randomfeat,ap_randomval,ap_rf_1,ap_rf_2,ap_rf_3,AP_AE_All,AP_VAE_All,AP_CAE_1_All,AP_CVAE_1_All,AP_CAE_2_All,AP_CVAE_2_All,AP_CAE_3_All,AP_CVAE_3_All])
print('')
print([REC_Mantel[0],rec_randomfeat[0],rec_randomval[0],rec_rf_1[0],rec_rf_2[0],rec_rf_3[0],REC_AE_ALL[0],REC_VAE_ALL[0],REC_CAE_1_ALL[0],REC_CVAE_1_ALL[0],REC_CAE_2_ALL[0],REC_CVAE_2_ALL[0],REC_CAE_3_ALL[0],REC_CVAE_3_ALL[0]])
print('')
print([REC_Mantel[2],rec_randomfeat[2],rec_randomval[2],rec_rf_1[2],rec_rf_2[2],rec_rf_3[2],REC_AE_ALL[2],REC_VAE_ALL[2],REC_CAE_1_ALL[2],REC_CVAE_1_ALL[2],REC_CAE_2_ALL[2],REC_CVAE_2_ALL[2],REC_CAE_3_ALL[2],REC_CVAE_3_ALL[2]])
print('')
print([REC_Mantel[4],rec_randomfeat[4],rec_randomval[4],rec_rf_1[4],rec_rf_2[4],rec_rf_3[4],REC_AE_ALL[4],REC_VAE_ALL[4],REC_CAE_1_ALL[4],REC_CVAE_1_ALL[4],REC_CAE_2_ALL[4],REC_CVAE_2_ALL[4],REC_CAE_3_ALL[4],REC_CVAE_3_ALL[4]])
print('')
print([np.mean(REC_Mantel[:4]),np.mean(rec_randomfeat[:4]),np.mean(rec_randomval[:4]),np.mean(rec_rf_1[:4]),np.mean(rec_rf_2[:4]),np.mean(rec_rf_3[:4]),np.mean(REC_AE_ALL[:4]),np.mean(REC_VAE_ALL[:4]),np.mean(REC_CAE_1_ALL[:4]),np.mean(REC_CVAE_1_ALL[:4]),np.mean(REC_CAE_2_ALL[:4]),np.mean(REC_CVAE_2_ALL[:4]),np.mean(REC_CAE_3_ALL[:4]),np.mean(REC_CVAE_3_ALL[:4])])






size = 150
size2 = 10

rec_rf_Mean = (rec_rf_1+rec_rf_2+rec_rf_3)/3
rec_rf_Best = rec_rf_3.copy()
REC_LEARNT_Mean = (REC_AE+REC_VAE+REC_CAE_1+REC_CAE_2+REC_CAE_3+REC_CVAE_1+REC_CVAE_2+REC_CVAE_3)/8
REC_LEARNT_Best = REC_CVAE_3_Final.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomfeat,'--',c='dimgray',label='Random Engineered',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomval,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,rec_rf_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,rec_rf_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg.')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

rec_rf_Mean = (rec_rf_1+rec_rf_2+rec_rf_3)/3
rec_rf_Best = rec_rf_3.copy()
REC_LEARNT_Mean = (REC_AE+REC_VAE+REC_CAE_1+REC_CAE_2+REC_CAE_3+REC_CVAE_1+REC_CVAE_2+REC_CVAE_3)/8
REC_LEARNT_Best = REC_CVAE_3.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomfeat,'--',c='dimgray',label='Random Engineered',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomval,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,rec_rf_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,rec_rf_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg.')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

rec_rf_Mean = (rec_rf_1+rec_rf_2+rec_rf_3)/3
rec_rf_Best = rec_rf_3.copy()
REC_LEARNT_Mean = (REC_AE_ALL+REC_VAE_ALL+REC_CAE_1_ALL+REC_CAE_2_ALL+REC_CAE_3_ALL+REC_CVAE_1_ALL+REC_CVAE_2_ALL+REC_CVAE_3_ALL)/8
REC_LEARNT_Best = REC_CVAE_3_ALL.copy()
REC_LEARNT_Best_Final = REC_CVAE_3_Final_ALL.copy()
#REC_LEARNT_Best = REC_CVAE_3_Final_ALL.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomfeat,'--',c='dimgray',label='Random Features',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,rec_randomval,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,rec_rf_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,rec_rf_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg. (TP-2)')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best (TP-2)')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best_Final,marker='D',edgecolor='black',s=size,c='red',label='Learnt Best (TP-3)')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(16,12))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'D--',c='black',label='Man',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,rec_randomfeat,'D--',c='dimgray',label='RnE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,rec_randomval,'D--',c='lightgray',label='RnV',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,rec_rf_1,'v-',c='orange',label='E1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,rec_rf_2,'^-',c='orange',label='E2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,rec_rf_3,'o-',c='orange',label='E3',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_AE,'o-',c='red',label='AE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_VAE,'o-',c='magenta',label='VAE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_1,'v-',c='green',label='CAE1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_2,'^-',c='green',label='CAE2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_3,'o-',c='green',label='CAE3',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_1,'v-',c='blue',label='CVAE1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_2,'^-',c='blue',label='CVAE2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_3,'o-',c='blue',label='CVAE3',linewidth=1,ms=8)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(12,9))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(9)+1,rec_rf_1[:9],'v-',c='orange',label='E1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,rec_rf_2[:9],'^-',c='orange',label='E2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,rec_rf_3[:9],'o-',c='orange',label='E3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_AE[:9],'o-',c='red',label='AE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_VAE[:9],'o-',c='magenta',label='VAE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_1[:9],'v-',c='green',label='CAE1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_2[:9],'^-',c='green',label='CAE2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_3[:9],'o-',c='green',label='CAE3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_1[:9],'v-',c='blue',label='CVAE1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_2[:9],'^-',c='blue',label='CVAE2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_3[:9],'o-',c='blue',label='CVAE3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_Mantel[:9],'D--',c='black',label='Man',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,rec_randomfeat[:9],'D--',c='dimgray',label='RnE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,rec_randomval[:9],'D--',c='lightgray',label='RnV',linewidth=1,ms=10)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 10, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(12,9))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(7)+1,rec_rf_1[:7],'v-',c='orange',label='E1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,rec_rf_2[:7],'^-',c='orange',label='E2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,rec_rf_3[:7],'o-',c='orange',label='E3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_AE[:7],'o-',c='red',label='AE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_VAE[:7],'o-',c='magenta',label='VAE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_1[:7],'v-',c='green',label='CAE1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_2[:7],'^-',c='green',label='CAE2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_3[:7],'o-',c='green',label='CAE3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_1[:7],'v-',c='blue',label='CVAE1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_2[:7],'^-',c='blue',label='CVAE2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_3[:7],'o-',c='blue',label='CVAE3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_Mantel[:7],'D--',c='black',label='Man',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,rec_randomfeat[:7],'D--',c='dimgray',label='RnE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,rec_randomval[:7],'D--',c='lightgray',label='RnV',linewidth=1,ms=10)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 8, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()
