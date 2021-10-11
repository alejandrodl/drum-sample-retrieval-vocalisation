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

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import StratifiedShuffleSplit

from networks import *


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=0, min_delta=0.):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
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
        if np.less(current, self.best) and self.best-current>=self.min_delta:
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
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Set the value back to the optimizer before this epoch starts
                K.set_value(self.model.optimizer.lr, lr/10)



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
modes = ['unsupervised_bark']

percentage_train = 80

epochs = 10000

batch_size = 512
#latent_dim = 16

#num_crossval = 5
num_iterations = 1

# Main

frame_size = '1024'
#lr = 1e-3
#kernel_height = 3
#kernel_width = 3

print('Loading dataset...')

print('VIPS')

Pretrain_Dataset_Eval_REF = np.load('../../data/interim/Dataset_VIPS_Ref_Bark.npy')
Pretrain_Classes_Eval_REF = np.load('../../data/interim/Classes_VIPS_Ref.npy')

Pretrain_Dataset_Eval_IMI = np.load('../../data/interim/Dataset_VIPS_Imi_Bark.npy')
Pretrain_Classes_Eval_IMI = np.load('../../data/interim/Classes_VIPS_Imi.npy')

print('Done.')

print('Normalising data...')

# Normalise data

print('Normalise')

min_data = -144.0
max_data = -37.0

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

    print('Transforming labels...')

    if 'RI' in mode:

        num_classes = 2

    elif 'KSH' in mode:

        num_classes = 4

    elif 'RI_KSH' in mode:

        num_classes = 8

    print('Done.')

    # Train models

    print('Training models...')

    for it in range(1):

        print('\n')
        print('Iteration ' + str(it))
        print('\n')

        validation_accuracy = -1
        validation_loss = np.inf

        set_seeds(it)

        if 'unsupervised' in mode:

            set_seeds(it)

            # Encoder

            encoder_input = keras.Input(shape=(128, 128, 1))

            x = layers.Conv2D(filters=1, kernel_size=(3,5), strides=(1,1), activation=None, padding='same')(encoder_input)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Flatten()(x)
            #x = layers.Dropout(0.3)(x)
            #x = layers.Dense(64, activation="relu")(x)

            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

            z = sampling([z_mean, z_log_var])

            encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

            # Decoder

            latent_inputs = keras.Input(shape=(latent_dim,))

            #x = layers.Dropout(0.5)(latent_inputs)
            dec = layers.Dense(units=4*4*128, activation="relu")(latent_inputs)
            #x = layers.Dropout(0.5)(x)
            dec = layers.Reshape(target_shape=(4,4,128))(dec)
            dec = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=1, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=1, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            dec = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation=None)(dec)
            dec = layers.BatchNormalization()(dec)
            dec = layers.ReLU()(dec)
            decoder_outputs = layers.Conv2DTranspose(1, (3,5), activation="relu", padding="same")(dec)

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

        else:

            set_seeds(it)

            # Encoder

            encoder_input = keras.Input(shape=(128, 128, 1))
            encoder_class = Input(shape=(num_classes,))

            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(encoder_input)
            #x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
            #x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.Dropout(0.3)(x)

            n_x_conv = x.shape
            x = Flatten()(x)
            n_x_flattened = x.shape[1]
            #x = Dense(128, activation='relu')(x)
            #x = layers.Dropout(0.3)(x)
            #x = Dense(latent_dim, activation='relu')(x)
            #x_encoded = Dense(latent_dim, activation='relu')(x)
            #mu = Dense(latent_dim, activation='linear')(x_encoded)
            #log_var = Dense(latent_dim, activation='linear')(x_encoded)
            mu = Dense(latent_dim, activation='linear')(x)
            log_var = Dense(latent_dim, activation='linear')(x)
            # encoder sampler
            z = Lambda(sample_z, output_shape=(latent_dim,))([mu, log_var])
            z_cond = Concatenate(axis=-1)([z, encoder_class])

            encoder = Model([encoder_input, encoder_class], [mu, log_var, z, encoder_class])

            # Decoder

            latent_inputs = Input(shape=(latent_dim,))
            decoder_classes = Input(shape=(num_classes,))

            decoder_input = Concatenate(axis=1)([latent_inputs, decoder_classes])

            '''dec = Dense(latent_dim, activation='relu')(decoder_input)
            dec = Dense(128, activation='relu')(dec)
            dec = Dense(n_x_flattened, activation='relu')(dec)
            dec = Reshape(tuple(n_x_conv[1:]))(dec)
            dec = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation='relu')(dec)'''

            #dec = Dense(latent_dim, activation='relu')(decoder_input)
            #dec = layers.Dropout(0.5)(decoder_input)
            dec = Dense(128, activation='relu')(decoder_input)
            #dec = layers.Dropout(0.5)(dec)
            dec = Dense(n_x_flattened, activation='relu')(dec)
            dec = Reshape(tuple(n_x_conv[1:]))(dec)
            dec = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same', activation='relu')(dec)
            dec = layers.Conv2DTranspose(filters=8, kernel_size=5, strides=2, padding='same', activation='relu')(dec)
            decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', activation='relu')(dec)

            #decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(dec)

            decoder =  Model([latent_inputs, decoder_classes], decoder_outputs, name="decoder")
            print(decoder.summary())

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

        model.load_weights('../../models/' + mode + '/pretrained_' + mode + '_' + str(it) + '.tf')

        # Compute processed features

        Pretrain_Dataset_Eval_REF_Expanded = np.expand_dims(Pretrain_Dataset_Eval_REF,axis=-1).astype('float32')
        Pretrain_Dataset_Eval_IMI_Expanded = np.expand_dims(Pretrain_Dataset_Eval_IMI,axis=-1).astype('float32')
            
        if 'unsupervised' in mode:

            embeddings_ref, _, _ = encoder(Pretrain_Dataset_Eval_REF_Expanded)
            embeddings_imi, _, _ = encoder(Pretrain_Dataset_Eval_IMI_Expanded)
            reconstructions_ref = model.predict(Pretrain_Dataset_Eval_REF_Expanded)
            reconstructions_imi = model.predict(Pretrain_Dataset_Eval_IMI_Expanded)

        else:

            Pretrain_Classes_Eval_REF_OneHot = np.zeros((len(Pretrain_Classes_Eval_REF),num_classes))
            for n in range(len(Pretrain_Classes_Eval_REF)):
                Pretrain_Classes_Eval_REF_OneHot[n,int(Pretrain_Classes_Eval_REF[n])] = 1

            Pretrain_Classes_Eval_IMI_OneHot = np.zeros((len(Pretrain_Classes_Eval_IMI),num_classes))
            for n in range(len(Pretrain_Classes_Eval_IMI)):
                Pretrain_Classes_Eval_IMI_OneHot[n,int(Pretrain_Classes_Eval_IMI[n])] = 1

            embeddings_ref, _, _, _ = encoder([Pretrain_Dataset_Eval_REF_Expanded,Pretrain_Classes_Eval_REF_OneHot])
            embeddings_imi, _, _, _ = encoder([Pretrain_Dataset_Eval_IMI_Expanded,Pretrain_Classes_Eval_IMI_OneHot])
            reconstructions_ref = model.predict([Pretrain_Dataset_Eval_REF_Expanded,Pretrain_Classes_Eval_REF_OneHot])
            reconstructions_imi = model.predict([Pretrain_Dataset_Eval_IMI_Expanded,Pretrain_Classes_Eval_IMI_OneHot])

        print(embeddings_ref.shape)
        print(embeddings_imi.shape)

        np.save('../../data/processed/' + mode + '/embeddings_bark_ref_' + mode + '_' + str(it), embeddings_ref)
        np.save('../../data/processed/' + mode + '/embeddings_bark_imi_' + mode + '_' + str(it), embeddings_imi)

        np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_ref_' + mode + '_' + str(it), reconstructions_ref)
        np.save('../../data/processed/reconstructions/' + mode + '/reconstructions_imi_' + mode + '_' + str(it), reconstructions_imi)

        np.save('../../data/processed/reconstructions/' + mode + '/originals_ref_' + mode + '_' + str(it), Pretrain_Dataset_Eval_REF_Expanded)
        np.save('../../data/processed/reconstructions/' + mode + '/originals_imi_' + mode + '_' + str(it), Pretrain_Dataset_Eval_IMI_Expanded)

        tf.keras.backend.clear_session()



