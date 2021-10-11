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



os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

latent_dim = 32
        
# Parameters

#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_logical_devices('GPU')
#print(gpus)

#modes = ['unsupervised','RI','KSH','RI_KSH']
modes = ['ae_50']

percentage_train = 75

epochs = 20

batch_size = 256
#latent_dim = 16

#num_crossval = 5
num_iterations = 1

# Main

frame_size = '1024'
#lr = 1e-3
#kernel_height = 3
#kernel_width = 3

# AVP Personal

# Vocal Imitations #

Pretrain_Dataset_IMI = np.zeros((1, 128, 128))
Pretrain_Classes_IMI = np.zeros(1)

print('AVP')

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP_Bark.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP_Bark.npy')))

# AVP Fixed Small

print('AVP Fixed')

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_AVP_Fixed_Bark.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_AVP_Fixed_Bark.npy')))

# LVT 2

print('LVT 2')

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_2_Bark.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_2_Bark.npy')))

# LVT 3

print('LVT 3')

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_LVT_3_Bark.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_LVT_3_Bark.npy')))

# Beatbox

print('Beatbox')

Pretrain_Dataset_IMI = np.vstack((Pretrain_Dataset_IMI, np.load('../../data/interim/Dataset_Beatbox_Bark.npy')))
Pretrain_Classes_IMI = np.concatenate((Pretrain_Classes_IMI, np.load('../../data/interim/Classes_Beatbox_Bark.npy')))

Pretrain_Dataset_IMI = Pretrain_Dataset_IMI[1:]
Pretrain_Classes_IMI = Pretrain_Classes_IMI[1:]

# Real Drums #

Pretrain_Dataset_REF = np.zeros((1, 128, 128))
Pretrain_Classes_REF = np.zeros(1)

# BFD

print('BFD')

Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_BFD_Bark.npy')))
Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_BFD_Bark.npy')))

# Misc

print('Misc')

Pretrain_Dataset_REF = np.vstack((Pretrain_Dataset_REF, np.load('../../data/interim/Dataset_Misc_Bark.npy')))
Pretrain_Classes_REF = np.concatenate((Pretrain_Classes_REF, np.load('../../data/interim/Classes_Misc_Bark.npy')))

Pretrain_Dataset_REF = Pretrain_Dataset_REF[1:]
Pretrain_Classes_REF = Pretrain_Classes_REF[1:]

# Evaluation (VIPS) #

print('VIPS')

Pretrain_Dataset_Eval_REF = np.load('../../data/interim/Dataset_VIPS_Ref_Bark.npy')
Pretrain_Classes_Eval_REF = np.load('../../data/interim/Classes_VIPS_Ref.npy')

Pretrain_Dataset_Eval_IMI = np.load('../../data/interim/Dataset_VIPS_Imi_Bark.npy')
Pretrain_Classes_Eval_IMI = np.load('../../data/interim/Classes_VIPS_Imi.npy')

print('Done.')

print('Normalising data...')

# Normalise data

print('Normalisation')

#all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI,Pretrain_Dataset_Eval_REF,Pretrain_Dataset_Eval_IMI))
all_datasets = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI))

min_data = np.min(all_datasets)
max_data = np.max(all_datasets)

Pretrain_Dataset_REF = (Pretrain_Dataset_REF-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_IMI = (Pretrain_Dataset_IMI-min_data)/(max_data-min_data+1e-16)
Pretrain_Dataset_Eval_REF = np.clip((Pretrain_Dataset_Eval_REF-min_data)/(max_data-min_data+1e-16),0,1)
Pretrain_Dataset_Eval_IMI = np.clip((Pretrain_Dataset_Eval_IMI-min_data)/(max_data-min_data+1e-16),0,1)

Pretrain_Dataset = np.vstack((Pretrain_Dataset_REF,Pretrain_Dataset_IMI)).astype('float32')

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

    num_classes = 8

    #cutoff_train = int((percentage_train/100)*Pretrain_Dataset.shape[0])

    Pretrain_Classes_OneHot = np.zeros((Pretrain_Dataset.shape[0],num_classes))
    for n in range(Pretrain_Dataset.shape[0]):
        Pretrain_Classes_OneHot[n,int(Pretrain_Classes[n])] = 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=m)
    for train_index, test_index in sss.split(Pretrain_Dataset, Pretrain_Classes):
        pretrain_dataset_train, pretrain_dataset_test = Pretrain_Dataset[train_index], Pretrain_Dataset[test_index]
        pretrain_classes_train, pretrain_classes_test = Pretrain_Classes_OneHot[train_index], Pretrain_Classes_OneHot[test_index]


    pretrain_dataset_train = pretrain_dataset_train.astype('float32')
    pretrain_classes_train = pretrain_classes_train.astype('float32')
    pretrain_dataset_test = pretrain_dataset_test.astype('float32')
    pretrain_classes_test = pretrain_classes_test.astype('float32')

    pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1)
    pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1)

    #pretrain_dataset_train = Pretrain_Dataset[:cutoff_train].astype('float32')
    #pretrain_classes_train = Pretrain_Classes[:cutoff_train].astype('float32')
    #pretrain_dataset_test = Pretrain_Dataset[cutoff_train:].astype('float32')
    #pretrain_classes_test = Pretrain_Classes[cutoff_train:].astype('float32')

    #max_index_train = pretrain_dataset_train.shape[0]-(pretrain_dataset_train.shape[0]%batch_size)
    #max_index_test = pretrain_dataset_test.shape[0]-(pretrain_dataset_test.shape[0]%batch_size)

    #max_index = Pretrain_Dataset.shape[0]-(Pretrain_Dataset.shape[0]%batch_size)

    #pretrain_dataset_train = Pretrain_Dataset[:max_index].astype('float32')
    #pretrain_classes_train = Pretrain_Classes[:max_index].astype('float32')

    #pretrain_dataset_train = Pretrain_Dataset[:max_index_train].astype('float32')
    #pretrain_classes_train = Pretrain_Classes[:max_index_train].astype('float32')
    #pretrain_dataset_test = Pretrain_Dataset[:max_index_test].astype('float32')
    #pretrain_classes_test = Pretrain_Classes[:max_index_test].astype('float32')

    #pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1)
    #pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1)

    # Train models

    print('Training models...')

    for it in range(5):

        print('\n')
        print('Iteration ' + str(it))
        print('\n')

        validation_accuracy = -1
        validation_loss = np.inf

        set_seeds(it)

        # Encoder

        encoder_input = keras.Input(shape=(128, 128, 1))

        x = layers.Conv2D(filters=8, kernel_size=(3,5), strides=(2,2), activation=None, padding='same')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=16, kernel_size=10, strides=(4,2), activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=24, kernel_size=10, strides=(4,2), activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=32, kernel_size=10, strides=(4,4), activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        #x = layers.Dropout(0.3)(x)
        #x = layers.Dense(64, activation="relu")(x)

        z = layers.Dense(latent_dim, name="z")(x)

        encoder = keras.Model(encoder_input, z, name="encoder")

        # Decoder

        latent_inputs = keras.Input(shape=(latent_dim,))

        #x = layers.Dropout(0.5)(latent_inputs)
        dec = layers.Dense(units=1*4*64, activation="relu")(latent_inputs)
        #x = layers.Dropout(0.5)(x)
        dec = layers.Reshape(target_shape=(1,4,64))(dec)
        dec = layers.UpSampling2D(size=(4, 4))(dec)
        dec = layers.Conv2D(filters=24, kernel_size=10, strides=1, padding='same', activation=None)(dec)
        dec = layers.BatchNormalization()(dec)
        dec = layers.ReLU()(dec)
        dec = layers.UpSampling2D(size=(4, 2))(dec)
        dec = layers.Conv2D(filters=16, kernel_size=10, strides=1, padding='same', activation=None)(dec)
        dec = layers.BatchNormalization()(dec)
        dec = layers.ReLU()(dec)
        dec = layers.UpSampling2D(size=(4, 2))(dec)
        dec = layers.Conv2D(filters=8, kernel_size=10, strides=1, padding='same', activation=None)(dec)
        dec = layers.BatchNormalization()(dec)
        dec = layers.ReLU()(dec)
        dec = layers.UpSampling2D(size=(2, 2))(dec)
        decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3,5), strides=1, padding='same', activation='relu')(dec)

        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        # Define VAE

        decoder_output = decoder(encoder(encoder_input))
        model = Model(encoder_input, decoder_output)

        # Loss

        cae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
        cae_loss *= 128*128

        model.add_loss(cae_loss)
        model.compile(optimizer='adam')

        log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tb_cb = TensorBoard(
            log_dir=log_dir, 
            profile_batch=0)

        es_cb = EarlyStopping(
            monitor='val_loss',
            verbose=True,
            patience=20,
            restore_best_weights=True)

        lr_cb = ReduceLROnPlateau(
            monitor='val_loss',
            verbose=True,
            patience=10)

        cb = [tb_cb, es_cb, lr_cb]

        #training_generator = BalancedDataGenerator(pretrain_dataset_train, pretrain_classes_train, datagen, batch_size=batch_size)

        training_generator = tf.data.Dataset.from_tensor_slices(({"input_1": pretrain_dataset_train, "input_2": pretrain_classes_train}))
        training_generator = training_generator.batch(batch_size, drop_remainder=True)

        validation_generator= tf.data.Dataset.from_tensor_slices(({"input_1": pretrain_dataset_test, "input_2": pretrain_classes_test}))
        validation_generator = validation_generator.batch(batch_size, drop_remainder=True)
        #validation_generator = BalancedDataGenerator(pretrain_dataset_test, pretrain_classes_test, datagen, batch_size=batch_size)
        #steps_per_epoch = training_generator.steps_per_epoch

        #training_generator = training_generator.filter(lambda x, y: tf.equal(tf.shape(x), batch_size))

        #for step, (x_batch_train, y_batch_train) in enumerate(training_generator):
            #print(x_batch_train.shape)
            #print(y_batch_train.shape)

        checkpoint_path = "checkpoints_adib_bark/iuerwgyugcp_" + mode + "_" + str(it) + ".ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        if os.path.isfile(checkpoint_path+".index"):
            model.load_weights(checkpoint_path)

        with tf.device(gpu_name):

            #model.compile(optimizer=optimizer)
            #history = model.fit(training_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, epochs=epochs, callbacks=cb, shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0
            history = model.fit(training_generator, validation_data=(validation_generator, None), epochs=epochs, callbacks=[EarlyStoppingAtMinLoss(10,0),LearningRateSchedulerCustom(5),cp_callback], shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0
            #history = model.fit(pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test, None), callbacks=cb, shuffle=True)  #  , callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0

        model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + str(it) + '.tf')

        Pretrain_Dataset_Eval_REF_Expanded = np.expand_dims(Pretrain_Dataset_Eval_REF,axis=-1).astype('float32')
        Pretrain_Dataset_Eval_IMI_Expanded = np.expand_dims(Pretrain_Dataset_Eval_IMI,axis=-1).astype('float32')

        Pretrain_Classes_Eval_REF_OneHot = np.zeros((len(Pretrain_Classes_Eval_REF),num_classes))
        for n in range(len(Pretrain_Classes_Eval_REF)):
            Pretrain_Classes_Eval_REF_OneHot[n,int(Pretrain_Classes_Eval_REF[n])] = 1

        Pretrain_Classes_Eval_IMI_OneHot = np.zeros((len(Pretrain_Classes_Eval_IMI),num_classes))
        for n in range(len(Pretrain_Classes_Eval_IMI)):
            Pretrain_Classes_Eval_IMI_OneHot[n,int(Pretrain_Classes_Eval_IMI[n])] = 1

        embeddings_ref = encoder(Pretrain_Dataset_Eval_REF_Expanded)
        embeddings_imi = encoder(Pretrain_Dataset_Eval_IMI_Expanded)
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