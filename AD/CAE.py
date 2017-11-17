# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:55:25 2017

@author: raminmh
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten,Reshape
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

batch_size = 20


filt1 = 16
filt2 = 16
filt3 = 16
latent_size = 32


x = Input(shape=(60, 200, 3))

conv1 = Conv2D(filt1, (5, 5), activation='relu', padding='same')(x)
poo1 = MaxPooling2D((2, 2), padding='same')(conv1)
conv2 = Conv2D(filt2, (3, 3), activation='relu', padding='same')(poo1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
conv3 = Conv2D(filt3, (3, 3), activation='relu', padding='same')(pool2)
encoded = MaxPooling2D((2, 2), padding='same')(conv3)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
flat = Flatten()(encoded)


hidden = Dense(latent_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))

hidden_decode = Dense(3200, activation='relu', kernel_regularizer=regularizers.l2(0.01))
output_shape = (batch_size, 8, 25, filt3)
decoder_reshape = Reshape(output_shape[1:])
dconv1 = Conv2D(filt3, (3, 3), activation='relu', padding='same')
upsamp1 = UpSampling2D((2, 2))
dconv2 = Conv2D(filt2, (3, 3), activation='relu', padding='same')
upsamp2 = UpSampling2D((2, 2))
dconv3 = Conv2D(filt1, (3, 1), activation='relu')
upsamp3 = UpSampling2D((2, 2))
decoded = Conv2D(3, (5, 5), activation='sigmoid', padding='same')


decoded00 = hidden(flat)
decoded0 = hidden_decode(decoded00)
decoder_resh = decoder_reshape(decoded0)
decoded1 = dconv1(decoder_resh)
decoded2 = upsamp1(decoded1)
decoded3 = dconv2(decoded2)
decoded4 = upsamp2(decoded3)
decoded5 = dconv3(decoded4)
decoded6 = upsamp3(decoded5)
decoded7 = decoded(decoded6)


autoencoder = Model(x, decoded7)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()


## encoder, from inputs to latent space
#
#
#encoder = Model(x, decoded00)
##encoder.summary()
#
## generator, from latent space to reconstructed inputs
#decoder_input = Input(shape=(latent_size,))
#decoded01 = hidden_decode(decoder_input)
#output_shape1 = (batch_size, 8, 25, filt3)
#decoder_reshape1 = Reshape(output_shape1[1:])(decoded01)
#decoded11 = dconv1(decoder_reshape1)
#decoded21 = upsamp1(decoded11)
#decoded31 = dconv2(decoded21)
#decoded41 = upsamp2(decoded31)
#decoded51 = dconv3(decoded41)
#decoded61 = upsamp3(decoded51)
#decoded71 = decoded(decoded61)
#generator = Model(decoder_input, decoded71)
##generator.summary()


# =============================================================================
# ##### Training
# =============================================================================


x_train = AD_Images[0:6499,:,:,:]
x_test = AD_Images[6500:7195,:,:,:]
y_test = AD_labels[0:695,:]
y_test = y_test*1000
#y_test = ( y_test - np.amin(y_test))/(np.amax(y_test) - np.amin(y_test))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


Predictions = autoencoder.predict(x_test)

#encoded_predictions = encoder.predict(x_test)
#encoded_Data = encoder.predict(x_train)


Predictions_Data = autoencoder.predict(x_train)


#plt.figure(figsize=(6, 6))
# =============================================================================
# plt.scatter(encoded_predictions[:,0], y_test)
# plt.colorbar()
# plt.show()
# =============================================================================
