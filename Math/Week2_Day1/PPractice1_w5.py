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
#import copy



# Importing the training set
#dataset_train = pd.read_csv('Elec_ON_Price_Train.csv')
dataset_train = pd.read_csv('Elec_Demand_train.csv')

training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, len(training_set)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

'''
x_train = x_train[0:1000, :]
y_train = y_train[0:1000]
'''
# Reshaping, 
# NOTE: Keras expect data in the form of [batch_size, Time_steps, sequence_length]
#             sometimes refer to as [batch_size, timesteps, units]
#             or in other word  [No_samples, No_datapoints_in_time, No_featuers]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# -----------------------------------------------------------------------------
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras

# Initialising the RNN
my_regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
my_regressor.add(LSTM(units = 30, return_sequences = True, 
                      input_shape = (x_train.shape[1], 1)))
my_regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#my_regressor.add(LSTM(units = 50, return_sequences = True))
#my_regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#my_regressor.add(LSTM(units = 50, return_sequences = True))
#my_regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
my_regressor.add(LSTM(units = 30))
my_regressor.add(Dropout(0.2))

# Adding the output layer
my_regressor.add(Dense(units = 1))

# Compiling the RNN
my_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', 
                     metrics = ['accuracy'])

# Fitting the RNN to the Training set
history = my_regressor.fit(x_train, y_train, epochs = 100,
                           batch_size = 32)


# save the final model
#my_regressor.save('Price.h5')
my_regressor.save('demand.h5')

# to load keras model
# my_regressor = keras.models.load_model("Price.h5")
# my_regressor = keras.models.load_model("demand.h5")
 
# -----------------------------------------------------------------------------
# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test = pd.read_csv('Elec_ON_Price_Test.csv')
dataset_test = pd.read_csv('Elec_Demand_test.csv')


real_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['HOEP'], dataset_test['HOEP']), axis = 0)
dataset_total = pd.concat((dataset_train['Demand'], dataset_test['Demand']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60+len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = my_regressor.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# Visualising the results
plt.plot(real_price, color = 'red', label = 'Real ON Electricity Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted ON Electricity Price')
plt.title('ON Electricity Price Prediction')
plt.xlabel('Time')
plt.ylabel('ON Electricity Price')
plt.legend()
plt.show()


# list all the data in history
print(history.history.keys())

# Plot the accuracy for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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






