# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:785].values
y = dataset.iloc[:, 0].values
X = X.reshape(X.shape[0],28,28,1).astype(float)

dataset = pd.read_csv('test.csv')
X2 = dataset.iloc[:, 0:784].values
X2 = X2.reshape(X2.shape[0],28,28,1).astype(float)
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.utils import np_utils

# normalize inputs from 0-255 to 0-1
X = X/ 255
X2 = X2 / 255
# one hot encode outputs
y = np_utils.to_categorical(y)
num_classes = y.shape[1]

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (28,28,1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# To save it from overfitting
classifier.add(Dropout(0.2))

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

classifier.fit(X,y,epochs = 10,verbose=1)
# Predict
pred=classifier.predict_classes(X2)

# submission file
np.savetxt(
		'submission.csv', 
		np.c_[range(1,len(pred)+1),pred], 
		delimiter = ',', 
		header = 'ImageId,Label', 
		comments = '', 
		fmt = '%d'
	)