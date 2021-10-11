import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

# Encoder

encoder_inputs = keras.Input(shape=(128, 128, 1))

x = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(encoder_inputs)
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

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder

latent_inputs = keras.Input(input_shape=(latent_dim,))

x = layers.Dense(units=4*4*128, activation="relu")(x)
x = layers.Reshape(target_shape=(4,4,128))(x)
x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(x)

decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Define VAE

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        


