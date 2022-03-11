import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, UpSampling2D, Concatenate

from utils import *



def adib(latent_dim):

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

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model



def adib_var(latent_dim):

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

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = sampling([z_mean, z_log_var])

    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

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
    decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3,5), strides=1, padding='same', activation="relu")(dec)

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

    return model



def adib_cond(latent_dim, num_classes):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))
    classes = Input(shape=(num_classes,))

    x = layers.Conv2D(filters=8, kernel_size=(3,5), strides=(2,2), activation=None, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=9, strides=(4,2), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=32, kernel_size=9, strides=(4,2), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=9, strides=(4,4), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)

    x_cond = Concatenate(axis=-1)([x, classes])
    z = Dense(latent_dim, activation='linear', name="z_layer")(x_cond)

    encoder = Model([encoder_input, classes], [z, classes], name="encoder")

    # Decoder

    latent_inputs = keras.Input(shape=(latent_dim,))

    decoder_input = Concatenate(axis=1)([latent_inputs, classes])

    dec = layers.Dense(units=1*4*64, activation="relu")(decoder_input)
    dec = layers.Reshape(target_shape=(1,4,64))(dec)
    dec = layers.Conv2DTranspose(filters=32, kernel_size=9, strides=(4,4), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=16, kernel_size=9, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=8, kernel_size=9, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=(3,5), strides=(2,2), padding='same', activation='relu')(dec)

    decoder =  Model([latent_inputs, classes], decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder([[encoder_input, classes]]))

    model = Model([encoder_input, classes], decoder_output)

    # Loss

    cae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    cae_loss *= 128*128

    model.add_loss(cae_loss)
    model.compile(optimizer='adam')

    return model





def adib_var_cond(latent_dim, num_classes):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1), name="input")
    classes = keras.Input(shape=(num_classes,), name="class")

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

    x_cond = Concatenate(axis=-1)([x, classes])

    z_mean = layers.Dense(latent_dim, name="z_mean")(x_cond)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_cond)

    z = sampling([z_mean, z_log_var])

    encoder = Model([encoder_input, classes], [z_mean, z_log_var, z, classes], name="encoder")

    # Decoder

    latent_inputs = keras.Input(shape=(latent_dim,))

    decoder_input = Concatenate(axis=1)([latent_inputs, classes])

    dec = layers.Dense(units=1*4*64, activation="relu")(decoder_input)
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
    decoder_output = layers.Conv2D(filters=1, kernel_size=(3,5), strides=1, padding='same', activation="relu")(dec)

    decoder =  Model([latent_inputs, classes], decoder_output, name="decoder")

    #pdb.set_trace()

    # Define VAE

    decoder_output = decoder([[encoder([[encoder_input, classes]])[2], classes]])
    model = Model([encoder_input, classes], decoder_output)

    # Loss

    reconstruction_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    reconstruction_loss *= 128*128

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=1)
    kl_loss = -0.5 * kl_loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model



















def adib_mp(latent_dim):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))

    x = layers.Conv2D(filters=8, kernel_size=(3,5), activation=None, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=16, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=24, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=32, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
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

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model




def adib_tc(latent_dim):

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
    dec = layers.Conv2DTranspose(filters=32, kernel_size=10, strides=(4,4), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=16, kernel_size=10, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=8, kernel_size=10, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=(3,5), strides=(2,2), padding='same', activation='relu')(dec)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder(encoder_input))
    model = Model(encoder_input, decoder_output)

    # Loss

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model





def adib_mp_tc(latent_dim):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))

    x = layers.Conv2D(filters=8, kernel_size=(3,5), activation=None, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=16, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=24, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=32, kernel_size=10, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
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
    dec = layers.Conv2DTranspose(filters=32, kernel_size=10, strides=(4,4), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=16, kernel_size=10, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.Conv2DTranspose(filters=8, kernel_size=10, strides=(4,2), padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=(3,5), strides=(2,2), padding='same', activation='relu')(dec)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder(encoder_input))
    model = Model(encoder_input, decoder_output)

    # Loss

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model



def adib_filt_2(latent_dim):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))

    x = layers.Conv2D(filters=8, kernel_size=(3,5), activation=None, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=16, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=32, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,4))(x)
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
    dec = layers.UpSampling2D(size=(4,4))(dec)
    dec = layers.Conv2D(filters=32, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=16, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=8, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(2,2))(dec)
    decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3,5), padding='same', activation='relu')(dec)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder(encoder_input))
    model = Model(encoder_input, decoder_output)

    # Loss

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model





def adib_timbre(latent_dim):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))

    x_1 = layers.Conv2D(filters=8, kernel_size=(65,3), activation=None, padding='same')(encoder_input)
    x_2 = layers.Conv2D(filters=8, kernel_size=(43,5), activation=None, padding='same')(encoder_input)
    x_3 = layers.Conv2D(filters=8, kernel_size=(21,7), activation=None, padding='same')(encoder_input)
    x_4 = layers.Conv2D(filters=8, kernel_size=(7,21), activation=None, padding='same')(encoder_input)
    x_5 = layers.Conv2D(filters=8, kernel_size=(5,43), activation=None, padding='same')(encoder_input)
    x_6 = layers.Conv2D(filters=8, kernel_size=(3,65), activation=None, padding='same')(encoder_input)

    x_tim = Concatenate(axis=-1)([x_1,x_2,x_3,x_4,x_5,x_6])
    x_tim = layers.Conv2D(filters=8, kernel_size=(1,1), activation=None, padding='same')(x_tim)

    x = layers.BatchNormalization()(x_tim)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=16, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=32, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,4))(x)
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
    dec = layers.UpSampling2D(size=(4,4))(dec)
    dec = layers.Conv2D(filters=32, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=16, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=8, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(2,2))(dec)
    decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3,5), padding='same', activation='relu')(dec)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder(encoder_input))
    model = Model(encoder_input, decoder_output)

    # Loss

    vae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    vae_loss *= 128*128

    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    return model




def adib_best_cond(latent_dim, num_classes):

    # Encoder

    encoder_input = keras.Input(shape=(128, 128, 1))
    classes = Input(shape=(num_classes,))

    x = layers.Conv2D(filters=8, kernel_size=(3,5), activation=None, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=16, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=32, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=9, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(4,4))(x)
    x = layers.Flatten()(x)

    x_cond = Concatenate(axis=-1)([x, classes])
    z = Dense(latent_dim, activation='linear', name="z_layer")(x_cond)

    encoder = Model([encoder_input, classes], [z, classes], name="encoder")

    # Decoder

    latent_inputs = keras.Input(shape=(latent_dim,))

    decoder_input = Concatenate(axis=1)([latent_inputs, classes])

    dec = layers.Dense(units=1*4*64, activation="relu")(decoder_input)
    #x = layers.Dropout(0.5)(x)
    dec = layers.Reshape(target_shape=(1,4,64))(dec)
    dec = layers.UpSampling2D(size=(4,4))(dec)
    dec = layers.Conv2D(filters=32, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=16, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(4,2))(dec)
    dec = layers.Conv2D(filters=8, kernel_size=9, padding='same', activation=None)(dec)
    dec = layers.BatchNormalization()(dec)
    dec = layers.ReLU()(dec)
    dec = layers.UpSampling2D(size=(2,2))(dec)
    decoder_outputs = layers.Conv2D(filters=1, kernel_size=(3,5), padding='same', activation='relu')(dec)

    decoder =  Model([latent_inputs, classes], decoder_outputs, name="decoder")

    # Define VAE

    decoder_output = decoder(encoder([[encoder_input, classes]]))

    model = Model([encoder_input, classes], decoder_output)

    # Loss

    cae_loss = keras.losses.mse(keras.layers.Flatten()(encoder_input),keras.layers.Flatten()(decoder_output))
    cae_loss *= 128*128

    model.add_loss(cae_loss)
    model.compile(optimizer='adam')

    return model