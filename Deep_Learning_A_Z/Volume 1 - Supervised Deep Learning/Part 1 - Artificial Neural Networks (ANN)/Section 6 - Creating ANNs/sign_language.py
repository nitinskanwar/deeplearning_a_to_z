#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:52:40 2019

@author: Nitin Kanwar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data from the npy files
X = np.load("/home/exx/nitin_scripts/deeplearning_a_to_z/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 6 - Creating ANNs/data/input/X.npy")
Y = np.load("/home/exx/nitin_scripts/deeplearning_a_to_z/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 6 - Creating ANNs/data/input/Y.npy")

plt.subplot(1,2,1)
plt.imshow(X[266])

# Split data into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

X_train_flatten = X_train.reshape(1546,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(516,X_test.shape[1]*X_test.shape[2])

# Define the network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(output_dim = 64, init= 'uniform', activation='relu', input_dim = 4096))
        classifier.add(Dense(output_dim = 32, init= 'uniform', activation='relu'))
        classifier.add(Dense(output_dim = 16, init= 'uniform', activation='relu'))
        classifier.add(Dense(output_dim = 10, init= 'uniform', activation='softmax'))
        opt = optimizers.Adam(lr=0.001)
        classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
        return(classifier)
        
classifier = build_classifier()
classifier.fit(X_train_flatten, Y_train, batch_size= 10, epochs= 500)
        
cv_classifier = KerasClassifier(build_fn = build_classifier, epochs = 200)
accuracies = cross_val_score(estimator = cv_classifier, X = X_train_flatten, y = Y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()        
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))