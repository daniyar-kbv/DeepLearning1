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

# Import as pandas dataframes
data = pd.read_csv('Cancer_data.csv')

# Get the x and y data
x = data.iloc[:, 2:]
y = pd.DataFrame(data['diagnosis'])


'''
# fix the nan -------------------------
# method 1: delete the rows
dum = ~np.isnan (data_analysis)
new_data = data_analysis[dum]

# Or get indeses
inds = np.where(np.isfinite(data_analysis[:,0]))
new_data = data_analysis[inds]

# method 2: replace nan with mean
inds2 = np.where(np.isnan(data_analysis[:,0]))

#new_data = data_analysis
new_data = np.copy(data_analysis)

mean_of_col = np.nanmean(new_data, axis=0)

new_data[inds2] = mean_of_col 
'''

#-- One hot encoder
from sklearn.preprocessing import LabelEncoder

# encode the target
label_encoder = LabelEncoder()
y_encode = label_encoder.fit_transform(y)

# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encode, test_size = 0.2,\
                                                    random_state = 0)

# size/shape of dataframe
n_samples = x_train.shape[0]
n_featuers = x_train.shape[1]


# Make the NN -----------------------------------------------------------------

# Importing the Keras libraries and packages
from keras.layers import Dense
from keras.models import Sequential

#from tensorflow.keras.layers import Dense

# define and initialize the model
my_classifier = Sequential()

# Adding the input layer AND the first hidden layer (Pay attention to this)
my_classifier.add(Dense(units = 20, kernel_initializer = 'uniform',
                        activation = 'tanh', input_dim = n_featuers))

# Adding the second hidden layer
my_classifier.add(Dense(units = 8, kernel_initializer = 'uniform',
                                                    activation = 'sigmoid'))

# Adding the last (output) layer
my_classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

# Compiling the ANN
my_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

#my_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#                      metrics = ['accuracy'])

#-- plot the model
#from keras.utils import plot_model
#plot_model(my_classifier, to_file='model.png', show_shapes=True)

# Fitting the ANN to the Training set
history = my_classifier.fit(x_train, y_train, validation_split=0.2,
                            batch_size = 10, epochs = 100)


# Make predictions
# Predicting the Test set results
y_pred_train = my_classifier.predict(x_train)
y_pred_train = (y_pred_train > 0.5)

# Predicting the Test set results
y_pred_test = my_classifier.predict(x_test)
y_pred_test = (y_pred_test > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

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




