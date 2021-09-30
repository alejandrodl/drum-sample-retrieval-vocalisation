import os
import pdb
import sys
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

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

from networks import *



class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = np.mean(logs.get("val_loss"))
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



class LearningRateSchedulerCustom(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=0):
        super(LearningRateSchedulerCustom, self).__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        current = np.mean(logs.get("val_loss"))
        print(lr)
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Set the value back to the optimizer before this epoch starts
                K.set_value(self.model.optimizer.lr, lr/10)



os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.nice(0)
gpu_name = '/GPU:0'
#gpu_name = '/device:CPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def sample_z(args):
    mu, log_var = args
    batch = K.shape(mu)[0]
    eps = K.random_normal(shape=(batch, latent_dim))
    return mu + K.exp(log_var / 2) * eps

# Create VAE

latent_dim = 16
        
# Parameters

#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_logical_devices('GPU')
#print(gpus)

#modes = ['unsupervised','RI','KSH','RI_KSH']
modes = ['RI_KSH']

percentage_train = 80

epochs = 10000
patience_lr = 2
patience_early = 5

batch_size = 256
#latent_dim = 16

#num_crossval = 5
num_iterations = 10

# Main

frame_size = '1024'
#lr = 1e-3
#kernel_height = 3
#kernel_width = 3

print('Loading dataset...')

# Vocal Imitations #

Pretrain_Dataset_IMI = np.zeros((1, 128, 128))
Pretrain_Classes_IMI = np.zeros(1)

# AVP Personal

#Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP.npy')))
#Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP.npy')))

# AVP Fixed Small

#Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP_Fixed.npy')))
#Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP_Fixed.npy')))

# LVT 2

#Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_2.npy')))
#Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_2.npy')))

# LVT 3

#Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_3.npy')))
#Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_3.npy')))

# Beatbox

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_Beatbox.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_Beatbox.npy')))

Pretrain_Dataset_IMI = Pretrain_Dataset_IMI[1:]
Pretrain_Classes_IMI = Pretrain_Classes_IMI[1:]

# Real Drums #

Pretrain_Dataset_REF = np.zeros((1, 128, 128))
Pretrain_Classes_REF = np.zeros(1)

# BFD

Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_BFD.npy')))
Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_BFD.npy')))

# Misc

#Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_Misc.npy')))
#Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_Misc.npy')))

Pretrain_Dataset_REF = Pretrain_Dataset_REF[1:]
Pretrain_Classes_REF = Pretrain_Classes_REF[1:]

# Evaluation (VIPS) #

Pretrain_Dataset_Eval_REF = np.load('../../data/interim/Dataset_VIPS_Ref.npy')
Pretrain_Classes_Eval_REF = np.load('../../data/interim/Classes_VIPS_Ref.npy')

Pretrain_Dataset_Eval_IMI = np.load('../../data/interim/Dataset_VIPS_Imi.npy')
Pretrain_Classes_Eval_IMI = np.load('../../data/interim/Classes_VIPS_Imi.npy')

print('Done.')

print('Normalising data...')

# Normalise data

#all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI,Pretrain_Dataset_Eval_REF,Pretrain_Dataset_Eval_IMI))
all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI))

min_data = np.min(all_datasets)
max_data = np.max(all_datasets)

Pretrain_Dataset_REF = (Pretrain_Dataset_REF-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_IMI = (Pretrain_Dataset_IMI-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_Eval_REF = np.clip((Pretrain_Dataset_Eval_REF-min_data)/(max_data-min_data+1e-16),0,1)
Pretrain_Dataset_Eval_IMI = np.clip((Pretrain_Dataset_Eval_IMI-min_data)/(max_data-min_data+1e-16),0,1)

Pretrain_Dataset_REF = np.log(Pretrain_Dataset_REF+1e-4)
Pretrain_Dataset_IMI = np.log(Pretrain_Dataset_IMI+1e-4)
Pretrain_Dataset_Eval_REF = np.log(Pretrain_Dataset_Eval_REF+1e-4)
Pretrain_Dataset_Eval_IMI = np.log(Pretrain_Dataset_Eval_IMI+1e-4)

#all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI,Pretrain_Dataset_Eval_REF,Pretrain_Dataset_Eval_IMI))
all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI))

min_data = np.min(all_datasets)
max_data = np.max(all_datasets)

Pretrain_Dataset_REF = (Pretrain_Dataset_REF-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_IMI = (Pretrain_Dataset_IMI-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_Eval_REF = np.clip((Pretrain_Dataset_Eval_REF-min_data)/(max_data-min_data+1e-16),0,1)
Pretrain_Dataset_Eval_IMI = np.clip((Pretrain_Dataset_Eval_IMI-min_data)/(max_data-min_data+1e-16),0,1)

Pretrain_Dataset = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI)).astype('float32')

np.random.seed(0)
np.random.shuffle(Pretrain_Dataset)

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

    print('Transforming labels...')

    if mode=='unsupervised':

        Pretrain_Classes_REF_Num = np.zeros(len(Pretrain_Classes_REF))
        for n in range(len(Pretrain_Classes_REF)):
            if Pretrain_Classes_REF[n]=='kd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='sd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='hhc':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='hho':
                Pretrain_Classes_REF_Num[n] = 0

        Pretrain_Classes_IMI_Num = np.zeros(len(Pretrain_Classes_IMI))
        for n in range(len(Pretrain_Classes_IMI)):
            if Pretrain_Classes_IMI[n]=='kd':
                Pretrain_Classes_IMI_Num[n] = 0
            elif Pretrain_Classes_IMI[n]=='sd':
                Pretrain_Classes_IMI_Num[n] = 0
            elif Pretrain_Classes_IMI[n]=='hhc':
                Pretrain_Classes_IMI_Num[n] = 0
            elif Pretrain_Classes_IMI[n]=='hho':
                Pretrain_Classes_IMI_Num[n] = 0

        Pretrain_Classes = np.concatenate((Pretrain_Classes_REF_Num,Pretrain_Classes_IMI_Num)).astype('float32')

    elif mode=='RI':

        Pretrain_Classes_REF_Num = np.zeros(len(Pretrain_Classes_REF))
        for n in range(len(Pretrain_Classes_REF)):
            if Pretrain_Classes_REF[n]=='kd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='sd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='hhc':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='hho':
                Pretrain_Classes_REF_Num[n] = 0

        Pretrain_Classes_IMI_Num = np.zeros(len(Pretrain_Classes_IMI))
        for n in range(len(Pretrain_Classes_IMI)):
            if Pretrain_Classes_IMI[n]=='kd':
                Pretrain_Classes_IMI_Num[n] = 1
            elif Pretrain_Classes_IMI[n]=='sd':
                Pretrain_Classes_IMI_Num[n] = 1
            elif Pretrain_Classes_IMI[n]=='hhc':
                Pretrain_Classes_IMI_Num[n] = 1
            elif Pretrain_Classes_IMI[n]=='hho':
                Pretrain_Classes_IMI_Num[n] = 1

        for n in range(len(Pretrain_Classes_Eval_REF)):
            if Pretrain_Classes_Eval_REF[n]=='kd':
                Pretrain_Classes_Eval_REF[n] = 0
            elif Pretrain_Classes_Eval_REF[n]=='sd':
                Pretrain_Classes_Eval_REF[n] = 0
            elif Pretrain_Classes_Eval_REF[n]=='hhc':
                Pretrain_Classes_Eval_REF[n] = 0
            elif Pretrain_Classes_Eval_REF[n]=='hho':
                Pretrain_Classes_Eval_REF[n] = 0

        for n in range(len(Pretrain_Classes_Eval_IMI)):
            if Pretrain_Classes_Eval_IMI[n]=='kd':
                Pretrain_Classes_Eval_IMI[n] = 1
            elif Pretrain_Classes_Eval_IMI[n]=='sd':
                Pretrain_Classes_Eval_IMI[n] = 1
            elif Pretrain_Classes_Eval_IMI[n]=='hhc':
                Pretrain_Classes_Eval_IMI[n] = 1
            elif Pretrain_Classes_Eval_IMI[n]=='hho':
                Pretrain_Classes_Eval_IMI[n] = 1

        Pretrain_Classes = np.concatenate((Pretrain_Classes_REF_Num,Pretrain_Classes_IMI_Num)).astype('float32')

    elif mode=='KSH':

        Pretrain_Classes_REF_Num = np.zeros(len(Pretrain_Classes_REF))
        for n in range(len(Pretrain_Classes_REF)):
            if Pretrain_Classes_REF[n]=='kd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='sd':
                Pretrain_Classes_REF_Num[n] = 1
            elif Pretrain_Classes_REF[n]=='hhc':
                Pretrain_Classes_REF_Num[n] = 2
            elif Pretrain_Classes_REF[n]=='hho':
                Pretrain_Classes_REF_Num[n] = 3

        Pretrain_Classes_IMI_Num = np.zeros(len(Pretrain_Classes_IMI))
        for n in range(len(Pretrain_Classes_IMI)):
            if Pretrain_Classes_IMI[n]=='kd':
                Pretrain_Classes_IMI_Num[n] = 0
            elif Pretrain_Classes_IMI[n]=='sd':
                Pretrain_Classes_IMI_Num[n] = 1
            elif Pretrain_Classes_IMI[n]=='hhc':
                Pretrain_Classes_IMI_Num[n] = 2
            elif Pretrain_Classes_IMI[n]=='hho':
                Pretrain_Classes_IMI_Num[n] = 3

        for n in range(len(Pretrain_Classes_Eval_REF)):
            if Pretrain_Classes_Eval_REF[n]=='kd':
                Pretrain_Classes_Eval_REF[n] = 0
            elif Pretrain_Classes_Eval_REF[n]=='sd':
                Pretrain_Classes_Eval_REF[n] = 1
            elif Pretrain_Classes_Eval_REF[n]=='hhc':
                Pretrain_Classes_Eval_REF[n] = 2
            elif Pretrain_Classes_Eval_REF[n]=='hho':
                Pretrain_Classes_Eval_REF[n] = 3

        for n in range(len(Pretrain_Classes_Eval_IMI)):
            if Pretrain_Classes_Eval_IMI[n]=='kd':
                Pretrain_Classes_Eval_IMI[n] = 0
            elif Pretrain_Classes_Eval_IMI[n]=='sd':
                Pretrain_Classes_Eval_IMI[n] = 1
            elif Pretrain_Classes_Eval_IMI[n]=='hhc':
                Pretrain_Classes_Eval_IMI[n] = 2
            elif Pretrain_Classes_Eval_IMI[n]=='hho':
                Pretrain_Classes_Eval_IMI[n] = 3

        Pretrain_Classes = np.concatenate((Pretrain_Classes_REF_Num,Pretrain_Classes_IMI_Num)).astype('float32')

    elif mode=='RI_KSH':

        Pretrain_Classes_REF_Num = np.zeros(len(Pretrain_Classes_REF))
        for n in range(len(Pretrain_Classes_REF)):
            if Pretrain_Classes_REF[n]=='kd':
                Pretrain_Classes_REF_Num[n] = 0
            elif Pretrain_Classes_REF[n]=='sd':
                Pretrain_Classes_REF_Num[n] = 1
            elif Pretrain_Classes_REF[n]=='hhc':
                Pretrain_Classes_REF_Num[n] = 2
            elif Pretrain_Classes_REF[n]=='hho':
                Pretrain_Classes_REF_Num[n] = 3

        Pretrain_Classes_IMI_Num = np.zeros(len(Pretrain_Classes_IMI))
        for n in range(len(Pretrain_Classes_IMI)):
            if Pretrain_Classes_IMI[n]=='kd':
                Pretrain_Classes_IMI_Num[n] = 4
            elif Pretrain_Classes_IMI[n]=='sd':
                Pretrain_Classes_IMI_Num[n] = 5
            elif Pretrain_Classes_IMI[n]=='hhc':
                Pretrain_Classes_IMI_Num[n] = 6
            elif Pretrain_Classes_IMI[n]=='hho':
                Pretrain_Classes_IMI_Num[n] = 7

        for n in range(len(Pretrain_Classes_Eval_REF)):
            if Pretrain_Classes_Eval_REF[n]=='kd':
                Pretrain_Classes_Eval_REF[n] = 0
            elif Pretrain_Classes_Eval_REF[n]=='sd':
                Pretrain_Classes_Eval_REF[n] = 1
            elif Pretrain_Classes_Eval_REF[n]=='hhc':
                Pretrain_Classes_Eval_REF[n] = 2
            elif Pretrain_Classes_Eval_REF[n]=='hho':
                Pretrain_Classes_Eval_REF[n] = 3

        for n in range(len(Pretrain_Classes_Eval_IMI)):
            if Pretrain_Classes_Eval_IMI[n]=='kd':
                Pretrain_Classes_Eval_IMI[n] = 4
            elif Pretrain_Classes_Eval_IMI[n]=='sd':
                Pretrain_Classes_Eval_IMI[n] = 5
            elif Pretrain_Classes_Eval_IMI[n]=='hhc':
                Pretrain_Classes_Eval_IMI[n] = 6
            elif Pretrain_Classes_Eval_IMI[n]=='hho':
                Pretrain_Classes_Eval_IMI[n] = 7

        Pretrain_Classes = np.concatenate((Pretrain_Classes_REF_Num,Pretrain_Classes_IMI_Num)).astype('float32')

        np.random.seed(0)
        np.random.shuffle(Pretrain_Classes)

    print('Done.')

    if mode!='unsupervised':

        num_classes = int(np.max(Pretrain_Classes)+1)

        Pretrain_Classes_OneHot = np.zeros((Pretrain_Dataset.shape[0],num_classes))
        for n in range(Pretrain_Dataset.shape[0]):
            Pretrain_Classes_OneHot[n,int(Pretrain_Classes[n])] = 1

        Pretrain_Classes = Pretrain_Classes_OneHot.copy()

    cutoff_train = int((percentage_train/100)*Pretrain_Dataset.shape[0])

    pretrain_dataset_train = Pretrain_Dataset[:cutoff_train].astype('float32')
    pretrain_classes_train = Pretrain_Classes[:cutoff_train].astype('float32')
    pretrain_dataset_test = Pretrain_Dataset[cutoff_train:].astype('float32')
    pretrain_classes_test = Pretrain_Classes[cutoff_train:].astype('float32')

    max_index_train = pretrain_dataset_train.shape[0]-(pretrain_dataset_train.shape[0]%batch_size)
    max_index_test = pretrain_dataset_test.shape[0]-(pretrain_dataset_test.shape[0]%batch_size)

    #max_index = Pretrain_Dataset.shape[0]-(Pretrain_Dataset.shape[0]%batch_size)

    #pretrain_dataset_train = Pretrain_Dataset[:max_index].astype('float32')
    #pretrain_classes_train = Pretrain_Classes[:max_index].astype('float32')
    pretrain_dataset_train = Pretrain_Dataset[:max_index_train].astype('float32')
    pretrain_classes_train = Pretrain_Classes[:max_index_train].astype('float32')
    pretrain_dataset_test = Pretrain_Dataset[:max_index_test].astype('float32')
    pretrain_classes_test = Pretrain_Classes[:max_index_test].astype('float32')

    pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1)
    pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1)

    # Train models

    print('Training models...')

    for it in range(num_iterations):

        print('\n')
        print('Iteration ' + str(it))
        print('\n')

        validation_accuracy = -1
        validation_loss = np.inf

        set_seeds(it)

        if mode=='unsupervised':

            set_seeds(it)

            # Encoder

            encoder_input = keras.Input(shape=(128, 128, 1))

            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(encoder_input)
            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation="relu")(x)

            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

            z = sampling([z_mean, z_log_var])

            encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

            # Decoder

            latent_inputs = keras.Input(shape=(latent_dim,))

            x = layers.Dense(units=4*4*128, activation="relu")(latent_inputs)
            x = layers.Reshape(target_shape=(4,4,128))(x)
            x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(x)

            decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

            decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

            # Define VAE

            decoder_output = decoder(encoder(encoder_input)[2])
            model = Model(encoder_input, decoder_output)

            # Loss

            reconstruction_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
            reconstruction_loss *= 128*128

            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=1)
            kl_loss = -0.5 * kl_loss
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            model.add_loss(vae_loss)
            model.compile(optimizer='adam')

            log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

            tb_cb = TensorBoard(
                log_dir=log_dir, 
                profile_batch=0)

            es_cb = EarlyStopping(
                monitor='val_loss',
                verbose=True,
                patience=10,
                restore_best_weights=True)

            lr_cb = ReduceLROnPlateau(
                monitor='val_loss',
                verbose=True,
                patience=5)

            cb = [tb_cb, es_cb, lr_cb]

            with tf.device(gpu_name):

                #model.compile(optimizer=optimizer)
                history = model.fit(pretrain_dataset_train, validation_split=0.20, batch_size=batch_size, epochs=epochs, callbacks=cb, shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0
                #history = model.fit(pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test, None), callbacks=cb, shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0

        else:

            set_seeds(it)

            # Encoder

            encoder_input = keras.Input(shape=(128, 128, 1))
            encoder_class = Input(shape=(num_classes,))

            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(encoder_input)
            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

            n_x_conv = x.shape
            x = Flatten()(x)
            n_x_flattened = x.shape[1]
            x = Dense(128, activation='relu')(x)
            x = Dense(latent_dim, activation='relu')(x)
            x_encoded = Dense(latent_dim, activation='relu')(x)
            mu = Dense(latent_dim, activation='linear')(x_encoded)
            log_var = Dense(latent_dim, activation='linear')(x_encoded)
            # encoder sampler
            z = Lambda(sample_z, output_shape=(latent_dim,))([mu, log_var])
            z_cond = Concatenate(axis=-1)([z, encoder_class])

            encoder = Model([encoder_input, encoder_class], [mu, log_var, z, encoder_class])

            # Decoder

            latent_inputs = Input(shape=(latent_dim,))
            decoder_classes = Input(shape=(num_classes,))

            decoder_input = Concatenate(axis=1)([latent_inputs, decoder_classes])

            dec = Dense(latent_dim, activation='relu')(decoder_input)
            dec = Dense(128, activation='relu')(dec)
            dec = Dense(n_x_flattened, activation='relu')(dec)
            dec = Reshape(tuple(n_x_conv[1:]))(dec)
            dec = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation='relu')(dec)

            #decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(dec)

            decoder =  Model([latent_inputs, decoder_classes], decoder_outputs, name="decoder")

            # Define VAE

            decoder_output = decoder(encoder([[encoder_input, encoder_class]])[2:])

            # Loss

            reconstruction_loss = keras.losses.binary_crossentropy(keras.layers.Flatten()(encoder_input), keras.layers.Flatten()(decoder_output))
            reconstruction_loss *= 128*128
            kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
            cvae_loss = reconstruction_loss + kl_loss

            # Create model

            model = Model([encoder_input, encoder_class], decoder_output)
            model.add_loss(cvae_loss)
            model.compile(optimizer='adam')

            '''# Logs and callbacks

            log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

            tb_cb = TensorBoard(
                log_dir=log_dir, 
                profile_batch=0)

            es_cb = EarlyStopping(
                monitor='val_loss',
                verbose=True,
                patience=10,
                restore_best_weights=True)

            lr_cb = ReduceLROnPlateau(
                monitor='val_loss',
                verbose=True,
                patience=5)

            cb = [tb_cb, es_cb, lr_cb]'''

            with tf.device(gpu_name):

                #model.compile(optimizer=optimizer)
                history = model.fit([pretrain_dataset_train,pretrain_classes_train], batch_size=batch_size, epochs=epochs, callbacks=[EarlyStoppingAtMinLoss(7),LearningRateSchedulerCustom(3)], shuffle=True, validation_data=([pretrain_dataset_test,pretrain_classes_test], None))  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0
                #history = model.fit([pretrain_dataset_train,pretrain_classes_train], validation_split=0.20, batch_size=batch_size, epochs=epochs, callbacks=[EarlyStoppingAtMinLoss(7),LearningRateSchedulerCustom(3)], shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0

        model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + str(it) + '.tf')

        # Compute processed features

        Pretrain_Dataset_Eval_REF_Expanded = np.expand_dims(Pretrain_Dataset_Eval_REF,axis=-1).astype('float32')
        Pretrain_Dataset_Eval_IMI_Expanded = np.expand_dims(Pretrain_Dataset_Eval_IMI,axis=-1).astype('float32')
            
        if mode=='unsupervised':

            embeddings_ref, _, _ = encoder(Pretrain_Dataset_Eval_REF_Expanded)
            embeddings_imi, _, _ = encoder(Pretrain_Dataset_Eval_IMI_Expanded)
            reconstructions_ref = model.predict(Pretrain_Dataset_Eval_REF_Expanded)
            reconstructions_imi = model.predict(Pretrain_Dataset_Eval_IMI_Expanded)

        else:

            embeddings_ref, _, _, _ = encoder(Pretrain_Dataset_Eval_REF_Expanded,Pretrain_Classes_Eval_REF_OneHot)
            embeddings_imi, _, _, _ = encoder(Pretrain_Dataset_Eval_IMI_Expanded,Pretrain_Classes_Eval_IMI_OneHot)
            reconstructions_ref = model.predict(Pretrain_Dataset_Eval_REF_Expanded,Pretrain_Classes_Eval_REF_OneHot)
            reconstructions_imi = model.predict(Pretrain_Dataset_Eval_IMI_Expanded,Pretrain_Classes_Eval_IMI_OneHot)

        print(embeddings_ref.shape)
        print(embeddings_imi.shape)

        np.save('../../data/processed/' + mode + '/embeddings_ref_' + mode + '_' + str(it), embeddings_ref)
        np.save('../../data/processed/' + mode + '/embeddings_imi_' + mode + '_' + str(it), embeddings_imi)

        np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_ref_' + mode + '_' + str(it), reconstructions_ref)
        np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_imi_' + mode + '_' + str(it), reconstructions_imi)

        np.save('../../data/processed/reconstructions/' + mode + '/originals_ref_' + mode + '_' + str(it), Pretrain_Dataset_Eval_REF_Expanded)
        np.save('../../data/processed/reconstructions/' + mode + '/originals_imi_' + mode + '_' + str(it), Pretrain_Dataset_Eval_IMI_Expanded)

        tf.keras.backend.clear_session()



