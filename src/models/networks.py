import pdb
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import tensorflow_probability as tfp


class VAE_Interim(tf.keras.Model):

    def __init__(self, latent_dim):
        super(VAE_Interim, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim+self.latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=4*4*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=2, padding='same')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar, batch_size):
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False): # apply_sigmoid=False
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, batch_size)
        out = self.decode(z)
        return out



class CVAE_Interim(tf.keras.Model):

    def __init__(self, latent_dim, num_labels):
        super(CVAE_Interim, self).__init__()
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten()
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=4*4*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=2, padding='same')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100,self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, labels):
        x = self.encoder(x)
        x = tf.concat([x,labels],1)
        x = tf.keras.layers.Dense(self.latent_dim+self.latent_dim)(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar, batch_size):
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False): # apply_sigmoid=False
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        labels = x[:,:self.num_labels,128,0]
        x = x[:,:,:128,:]
        mean, logvar = self.encode(x, labels)
        z = self.reparameterize(mean, logvar, batch_size)
        out = self.decode(z)
        return out



'''class CVAE_Interim(tf.keras.Model):

    def __init__(self, latent_dim, num_labels):
        super(CVAE_Interim, self).__init__()
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=4*4*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(kernel_height,kernel_width), strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=(kernel_height,kernel_width), strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(kernel_height,kernel_width), strides=2, padding='same'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, labels):
        x = self.encoder(x)
        x = tf.concat([x, labels], 0)
        x = tf.keras.layers.Dense(self.latent_dim+self.latent_dim)(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar, batch_size):
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False): # apply_sigmoid=False
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        labels = x[:,:self.num_labels,128,0]
        x = x[:,:,:128,:]
        mean, logvar = self.encode(x, labels)
        z = self.reparameterize(mean, logvar, batch_size)
        out = self.decode(z)
        return out'''


