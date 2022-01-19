# Different ways to do the same stuff
# pay attention to the loss function

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing, model_selection
from keras.models import Sequential 
from keras.layers import Dense


data = pd.read_csv('Iris.csv')
data = data.drop(['Id'], axis =1) #drop id col

data = shuffle(data)

# look at first 8 entries as predictions
i = 8
data_to_predict = data[:i].reset_index(drop = True)
predict_species = data_to_predict.Species 
predict_species = np.array(predict_species)
prediction = np.array(data_to_predict.drop(['Species'],axis= 1))

# remove 8 predictions from out train data
data = data[i:].reset_index(drop = True) 

X = data.drop(['Species'], axis = 1)
X = np.array(X)
Y = data['Species']

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y) # Converts a class vector (integers) to binary class matrix like OHE

train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0) # shuffles train & test

input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 10, batch_size = 2)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(prediction) # model predictions
prediction_ = np.argmax(to_categorical(predictions), axis = 1)  # reverse 
prediction_ = encoder.inverse_transform(prediction_)            # vectorization

for i, j in zip(prediction_ , predict_species):
    print( " the nn predict {}, and the species to find is {}".format(i,j))