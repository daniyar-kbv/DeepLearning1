# %% import libaries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing, model_selection
from keras.models import Sequential, load_model 
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

# %% Preprocess

# import data
data = pd.read_csv('drug200.csv')

# Nulls?
data.isnull().sum()

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

# Standardize feature(s)?

# Encode feature(s)?
data.info()

Sex_encoder = LabelEncoder()
Sex_encoder.fit(X["Sex"])
X["Sex"] = Sex_encoder.transform(X["Sex"])

BP_encoder = LabelEncoder()
BP_encoder.fit(X["BP"])
X["BP"] = BP_encoder.transform(X["BP"])

Cholesterol_encoder = LabelEncoder()
Cholesterol_encoder.fit(X["Cholesterol"])
X["Cholesterol"] = Cholesterol_encoder.transform(X["Cholesterol"])

Y_encoder = LabelEncoder()
Y_encoder.fit(Y)
Y = Y_encoder.transform(Y)
#Y = np_utils.to_categorical(Y)

# split data
train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 0)

# %% NN architecture

#input_dim = train_x.shape[1]

model = Sequential()
model.add(Dense(8, input_dim = 5 , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 10, batch_size = 2)

# %% NN training

model.summary() # model architecture
checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True) # save model's best weights

history = model.fit(
                    train_x, 
                    train_y, 
                    epochs=30, 
                    batch_size=30, 
                    callbacks = [checkpointer], 
                    validation_data = (test_x, test_y)
                    )

model_name = "drug.h5"
model.save(model_name)
model = load_model(model_name)

# %% Predictions

y_pred = model.predict(test_x)
y_pred = Y_encoder.inverse_transform(y_pred)

Y = Y_encoder.inverse_transform(Y)

cm = confusion_matrix(y_pred, test_y)

print("\nHistory Keys:\n")
print(history.history.keys())

plt.subplots() # open a new plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

plt.subplots() # open a new plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()