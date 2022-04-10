import os
import pdb
import sys
import math
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa
from itertools import combinations
from tensorflow.keras import layers
#import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
#tf.config.experimental_run_functions_eagerly(True)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import Sequence
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss

from sklearn.model_selection import StratifiedShuffleSplit

from networks import *
from utils import *



os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Create VAE

modes = ['adib']
#submodes = ['mp','tc','mp_tc','filt_1','filt_2','filt_3','deep_1','deep_2','deep_3','heuristic']
submodes = ['heuristic']

latent_dim = 32

epochs = 200

batch_size = 512
num_iterations = 5

replacements_0 = {'kd':0,'sd':0,'hhc':0,'hho':0}
replacements_1 = {'kd':1,'sd':1,'hhc':1,'hho':1}
replacements_03 = {'kd':0,'sd':1,'hhc':2,'hho':3}
replacements_47 = {'kd':4,'sd':5,'hhc':6,'hho':7}

# Vocal Imitations

Pretrain_Dataset_IMI = np.zeros((1, 128, 128)).astype('float32')
Pretrain_Classes_IMI = np.zeros(1)

print('AVP')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP.npy')))

print('AVP Fixed')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP_Fixed.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP_Fixed.npy')))

print('LVT 2')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_2.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_2.npy')))

print('LVT 3')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_3.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_3.npy')))

print('BTX')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_BTX.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_BTX.npy')))

print('Beatbox')
Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_Beatbox.npy').astype('float32')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_Beatbox.npy')))

Pretrain_Dataset_IMI = Pretrain_Dataset_IMI[1:].astype('float32')
Pretrain_Classes_IMI = Pretrain_Classes_IMI[1:]

# Real Drums

Pretrain_Dataset_REF = np.zeros((1, 128, 128)).astype('float32')
Pretrain_Classes_REF = np.zeros(1)

print('BFD')
Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_BFD.npy').astype('float32')))
Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_BFD.npy')))

print('Misc')
Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_Misc.npy').astype('float32')))
Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_Misc.npy')))

Pretrain_Dataset_REF = Pretrain_Dataset_REF[1:].astype('float32')
Pretrain_Classes_REF = Pretrain_Classes_REF[1:]

# Evaluation (VIPS)

print('VIPS')
Pretrain_Dataset_Eval_REF = np.load('../../data/interim/Dataset_VIPS_Ref.npy').astype('float32')
Pretrain_Classes_Eval_REF = np.load('../../data/interim/Classes_VIPS_Ref.npy')
Pretrain_Dataset_Eval_IMI = np.load('../../data/interim/Dataset_VIPS_Imi.npy').astype('float32')
Pretrain_Classes_Eval_IMI = np.load('../../data/interim/Classes_VIPS_Imi.npy')

print('Done.')

print('Normalising data...')

all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI))

print('Normalising data...')

min_data = np.min(all_datasets)
max_data = np.max(all_datasets)

print('Normalising data...')

Pretrain_Dataset_REF = (Pretrain_Dataset_REF-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_IMI = (Pretrain_Dataset_IMI-min_data)/(max_data-min_data+1e-16)
print('Normalising data...')
Pretrain_Dataset_Eval_REF = np.clip((Pretrain_Dataset_Eval_REF-min_data)/(max_data-min_data+1e-16),0,1)
Pretrain_Dataset_Eval_IMI = np.clip((Pretrain_Dataset_Eval_IMI-min_data)/(max_data-min_data+1e-16),0,1)

print('Normalising data...')

Pretrain_Dataset = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI)).astype('float32')

print('Done.')

# Main loop

for m in range(len(modes)):

    for sm in range(len(submodes)):

        mode = modes[m]
        submode = submodes[sm]

        mode = mode + '_' + submode

        if not os.path.isdir('../../models/' + mode):
            os.mkdir('../../models/' + mode)

        if not os.path.isdir('../../data/processed/' + mode):
            os.mkdir('../../data/processed/' + mode)

        if not os.path.isdir('../../data/processed/reconstructions/' + mode):
            os.mkdir('../../data/processed/reconstructions/' + mode)

        print('\n')
        print(mode)
        print('\n')

        replacer = replacements_0.get
        Pretrain_Classes_REF_Num = np.array([replacer(n,n) for n in Pretrain_Classes_REF.astype('U13')])
        Pretrain_Classes_IMI_Num = np.array([replacer(n,n) for n in Pretrain_Classes_IMI.astype('U13')])
        Pretrain_Classes_Eval_REF = np.array([replacer(n,n) for n in Pretrain_Classes_Eval_REF.astype('U13')])
        Pretrain_Classes_Eval_IMI = np.array([replacer(n,n) for n in Pretrain_Classes_Eval_IMI.astype('U13')])

        Pretrain_Classes = np.concatenate((Pretrain_Classes_REF_Num,Pretrain_Classes_IMI_Num))

        # Train models

        print('Training models...')

        for it in range(num_iterations):

            print('\n')
            print('Iteration ' + str(it))
            print('\n')

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=it)
            for train_index, test_index in sss.split(Pretrain_Dataset, Pretrain_Classes):
                pretrain_dataset_train, pretrain_dataset_test = Pretrain_Dataset[train_index], Pretrain_Dataset[test_index]
                pretrain_classes_train, pretrain_classes_test = Pretrain_Classes[train_index], Pretrain_Classes[test_index]

            pretrain_dataset_train = pretrain_dataset_train
            pretrain_classes_train = pretrain_classes_train
            pretrain_dataset_test = pretrain_dataset_test
            pretrain_classes_test = pretrain_classes_test

            pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1)
            pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1)

            validation_accuracy = -1
            validation_loss = np.inf

            set_seeds(it)

            if 'mp' in mode:
                model = adib_mp(latent_dim)
            elif 'tc' in mode:
                model = adib_tc(latent_dim)
            elif 'mp_tc' in mode:
                model = adib_mp_tc(latent_dim)
            elif 'filt_1' in mode:
                model = adib_filt_1(latent_dim)
            elif 'filt_2' in mode:
                model = adib_filt_2(latent_dim)
            elif 'filt_3' in mode:
                model = adib_filt_3(latent_dim)
            elif 'deep_1' in mode:
                model = adib_deep_1(latent_dim)
            elif 'deep_2' in mode:
                model = adib_deep_2(latent_dim)
            elif 'deep_3' in mode:
                model = adib_deep_3(latent_dim)
            elif 'heuristic' in mode:
                model = adib_timbre(latent_dim)
            
            training_generator = tf.data.Dataset.from_tensor_slices((pretrain_dataset_train, pretrain_classes_train))
            training_generator = training_generator.batch(batch_size, drop_remainder=True)

            validation_generator = tf.data.Dataset.from_tensor_slices((pretrain_dataset_test, pretrain_classes_test))
            validation_generator = validation_generator.batch(batch_size, drop_remainder=True)

            with tf.device(gpu_name):
                history = model.fit(training_generator, validation_data=validation_generator, epochs=epochs, callbacks=[EarlyStoppingAtMinLoss(7,0),LearningRateSchedulerCustom(4)], shuffle=True)

            model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + str(it) + '.tf')

            # Compute processed features

            Pretrain_Dataset_Eval_REF_Expanded = np.expand_dims(Pretrain_Dataset_Eval_REF,axis=-1)
            Pretrain_Dataset_Eval_IMI_Expanded = np.expand_dims(Pretrain_Dataset_Eval_IMI,axis=-1)

            extractor = tf.keras.Sequential()
            for layer in model.layers[:2]:
                extractor.add(layer)
            extractor.built = True

            embeddings_ref = extractor.predict(Pretrain_Dataset_Eval_REF_Expanded)
            embeddings_imi = extractor.predict(Pretrain_Dataset_Eval_IMI_Expanded)
            reconstructions_ref = model.predict(Pretrain_Dataset_Eval_REF_Expanded)
            reconstructions_imi = model.predict(Pretrain_Dataset_Eval_IMI_Expanded)

            print(embeddings_ref.shape)
            print(embeddings_imi.shape)

            np.save('../../data/processed/' + mode + '/embeddings_ref_' + mode + '_' + str(it), embeddings_ref)
            np.save('../../data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it), embeddings_imi)

            np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_ref_' + mode + '_' + str(it), reconstructions_ref)
            np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_imi_' + mode + '_' + str(it), reconstructions_imi)

            np.save('../../data/processed/reconstructions/' + mode + '/originals_ref_' + mode + '_' + str(it), Pretrain_Dataset_Eval_REF_Expanded)
            np.save('../../data/processed/reconstructions/' + mode + '/originals_imi_' + mode + '_' + str(it), Pretrain_Dataset_Eval_IMI_Expanded)

            tf.keras.backend.clear_session()