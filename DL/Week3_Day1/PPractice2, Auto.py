# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
#from keras.datasets import mnist

#import copy

# Import the dataset
#fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test_all, y_test_all) = fashion_mnist.load_data()

img_size=28 # width and height of image

# give name to classes
class_titles = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# pick a subset of data only, to save time
x_train = x_train_all[0:1000, :, :]
y_train = y_train_all[0:1000]
x_test = x_test_all[0:200, :, :]
y_test = y_test_all[0:200]

# pick a second subset for testing
x_test2 = x_test_all[-500:, :, :]
y_test2 = y_test_all[-500:]
                
# scale the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# size/shape of dataframe
n_samples = x_train.shape[0]

# look at some of the images
image_No = 100
plt.figure()
plt.imshow(x_test[image_No])

# look at some more
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.xlabel(class_titles[y_test[i]])
plt.show()


# Lets add noise to the images
mult_fact = 0.5
x_train_noisy = x_train + mult_fact * np.random.normal(size=x_train.shape) 
x_test_noisy = x_test + mult_fact * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

# clip to the range of 0 to 1 (negatives --> 0 and all bigger that 1 become 1)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# look at some of the noisy images
image_No = 100
plt.figure()
plt.imshow(x_train_noisy[image_No])

# look at some more
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test_noisy[i])
    plt.xlabel(class_titles[y_test[i]])
plt.show()



# the CNN Autoencoder --------------------------------------------------------------

# reshape images to [no_samples, width, height, No_channels]
#         No_channel reffer to color channels (in this example 1, they are grayscale)
x_train = x_train.reshape((x_train.shape[0], img_size, img_size, 1))
x_test = x_test.reshape((x_test.shape[0], img_size, img_size, 1))
x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0], img_size, img_size, 1))
x_test_noisy = x_test_noisy.reshape((x_test_noisy.shape[0], img_size, img_size, 1))


# Built the CNN Autoencoder
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# size of input
input_img = Input(shape=(img_size, img_size, 1))  

# Decoder
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)

# here is the bottleneck the size is (7, 7, 32)
encoding_dim = 7

# Encoder
decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)



# decoder and encoder together
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# the encoder part only
encoder = Model(input_img, encoded)
encoder.summary()

'''
# the decoder part only
# size input to encoder
encoded_input = Input(shape=(encoding_dim,))
# get the last layer of the autoencoder
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
'''

# Compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# fit
autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# predict results
decoded_imgs = autoencoder.predict(x_test_noisy)

# look at some more

plt.figure(figsize=(10,10))
for i in range(25):
    decoded_plot = decoded_imgs[i].reshape(img_size, img_size)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(decoded_plot)
    plt.xlabel(class_titles[y_test[i]])
plt.show()


# lets try encode and decode some digits
encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

plt.figure(figsize=(10,10))
for i in range(25):
    encoded_plot = encoded_imgs[i,:,:, 30].reshape(encoding_dim, encoding_dim)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(encoded_plot)
    plt.xlabel(class_titles[y_test[i]])
plt.show()



