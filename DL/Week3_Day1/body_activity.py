# Info:

#-- This code build a RNN to recognize human activity signals (from 6 activity)

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

#------------------------------------------------------------------------------
# based on different sensor data, we want to figure out if our subject is walking, siting, standing, etc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.utils import to_categorical
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

# Importing the training and test sets
data_main = list()

dum_data = pd.read_csv('total_acc_x_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('total_acc_y_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('total_acc_z_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)

dum_data = pd.read_csv('body_acc_x_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('body_acc_y_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('body_acc_z_train.txt', header=None, delim_whitespace=True)

data_main.append(dum_data)
dum_data = pd.read_csv('body_gyro_x_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('body_gyro_y_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)
dum_data = pd.read_csv('body_gyro_z_train.txt', header=None, delim_whitespace=True)
data_main.append(dum_data)

# make it a 3D aray with featuers, same rows and col as dum_data, and we have 9 dum_data
data = np.dstack(data_main)

# read the target values
y_data = pd.read_csv('y_train.txt', header=None, delim_whitespace=True)

# Train and test set data
x_train = data[0:6000, :, :]
y_train = y_data [0:6000]

x_test = data[6000:, :, :]
y_test = y_data [6000:]

# one hot encode the y train data
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

# delete category zero, no such category
y_train_hot = y_train_hot[:, 1:7]
y_test_hot = y_test_hot[:, 1:7]

# get important sizes and dimensions
n_sample = x_train.shape[0]
time_steps = x_train.shape[1]
n_features = x_train.shape[2]
n_outputs = y_train_hot.shape[1]

# Visualising some of the signal
test_sample = 150
data_plot = x_train[test_sample, :, 8]
plt.plot(data_plot)

plt.title(' title ')
plt.xlabel('Time')
plt.ylabel('signal value')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras

# Initialising the RNN
my_classifier = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
my_classifier.add(LSTM(units = 20, return_sequences = True,
                       input_shape = (time_steps, n_features)))
my_classifier.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
my_classifier.add(LSTM(units = 30, return_sequences = True))
my_classifier.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
my_classifier.add(LSTM(units = 20, return_sequences = True))
my_classifier.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
my_classifier.add(LSTM(units = 20))
my_classifier.add(Dropout(0.2))

# Adding the output layer
my_classifier.add(Dense(units = n_outputs, activation='softmax'))

# Compiling the RNN
my_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                     metrics = ['accuracy'])

my_classifier.summary()

# Fitting the RNN to the Training set
history = my_classifier.fit(x_train, y_train_hot, epochs = 10,
                           batch_size = 32, validation_data = (x_test, y_test_hot))

# save the final model
my_classifier.save('sig_classifier.h5')

# to load keras model
from keras.models import load_model
my_classifier = keras.models.load_model("sig_classifier.h5")
 
# evaluate model
#_, accuracy = my_classifier.evaluate(x_test, y_test_hot, batch_size=32)


# -----------------------------------------------------------------------------
# Making the predictions and visualising the results
y_test_pred_hot = my_classifier.predict(x_test)

#y_test_pred_hot [ y_test_pred_hot>0.5] = 1
#y_test_pred_hot [ y_test_pred_hot<0.5] = 0

# inverse the "to_categorical"
y_test_pred = np.argmax(y_test_pred_hot, axis=1) + 1


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)

# list all the data in history
print(history.history.keys())

# Plot the accuracy for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Plot the loss for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


