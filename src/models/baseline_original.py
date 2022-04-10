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

from __networks import *
from __utils import *



os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

modes = ['adib_original_data']
latent_dim = 32

epochs = 200

batch_size = 512
num_iterations = 5

print('Loading data...')

Pretrain_Dataset = np.load('../../data/interim/Dataset_Adib.npy').astype('float32')

Pretrain_Dataset_Eval_REF = np.load('../../data/interim/Dataset_VIPS_Original_Ref.npy').astype('float32')
Pretrain_Classes_Eval_REF = np.load('../../data/interim/Classes_VIPS_Original_Ref.npy')
Pretrain_Dataset_Eval_IMI = np.load('../../data/interim/Dataset_VIPS_Original_Imi.npy').astype('float32')
Pretrain_Classes_Eval_IMI = np.load('../../data/interim/Classes_VIPS_Original_Imi.npy')

print('Normalising data...')

min_data = -144.0
max_data = -33.162918

Pretrain_Dataset_Eval_REF = np.clip((Pretrain_Dataset_Eval_REF-min_data)/(max_data-min_data+1e-16),0,1)
Pretrain_Dataset_Eval_IMI = np.clip((Pretrain_Dataset_Eval_IMI-min_data)/(max_data-min_data+1e-16),0,1)

print('Done.')

# Main loop

for m in range(len(modes)):

    mode = modes[m]

    if not os.path.isdir('../../models/' + mode):
        os.mkdir('../../models/' + mode)

    if not os.path.isdir('../../data/processed/' + mode):
        os.mkdir('../../data/processed/' + mode)

    if not os.path.isdir('../../data/processed/reconstructions/' + mode):
        os.mkdir('../../data/processed/reconstructions/' + mode)

    print('\n')
    print(mode)
    print('\n')

    Pretrain_Classes = np.zeros(Pretrain_Dataset.shape[0])

    # Train models

    print('Training models...')

    for it in range(0,2):

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

        model = adib(latent_dim)

        training_generator = tf.data.Dataset.from_tensor_slices((pretrain_dataset_train, pretrain_classes_train))
        training_generator = training_generator.batch(batch_size, drop_remainder=True)

        validation_generator = tf.data.Dataset.from_tensor_slices((pretrain_dataset_test, pretrain_classes_test))
        validation_generator = validation_generator.batch(batch_size, drop_remainder=True)

        with tf.device(gpu_name):
            history = model.fit(training_generator, validation_data=validation_generator, epochs=epochs, callbacks=[EarlyStoppingAtMinLoss(10,0)], shuffle=True)

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