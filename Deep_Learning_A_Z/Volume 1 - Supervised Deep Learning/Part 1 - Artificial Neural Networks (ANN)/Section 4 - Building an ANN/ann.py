# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN
# Importing Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu', input_dim = 11))

# Adding second layer 
classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu'))

# Adding third layer 
classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu'))

# Adding the output layer 
classifier.add(Dense(output_dim = 1, init ='uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN
classifier.fit(X_train, y_train, batch_size= 1, epochs= 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework - Single observation how to resolve
data = [[0,0,600,1,40,3,60000,2,1,1,50000]]
data = sc.transform(data)
new_prediction = classifier.predict(data)
new_prediction = (new_prediction > 0.5)

#-----------------------------------------------------------------------------------------------------------------------------------------#

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
        # Initializing the ANN
        classifier = Sequential()
        
        # Adding input layer and first hidden layer
        classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu', input_dim = 11))
        
        # Adding second layer 
        classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu'))
        
        # Adding the output layer 
        classifier.add(Dense(output_dim = 1, init ='uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy']) 

        return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size= 10, epochs= 100) 
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

###################################################################################################
# Improving and Tuning the ANN with dropout
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu', input_dim = 11))

# Adding dropout to first hidden layer
classifier.add(Dropout(p = 0.1))

# Adding second layer 
classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu'))

# Adding dropout to first hidden layer
classifier.add(Dropout(p = 0.1))

# Adding the output layer 
classifier.add(Dense(output_dim = 1, init ='uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

################################### GRID SEARCH #############################################
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
        # Initializing the ANN
        classifier = Sequential()
        
        # Adding input layer and first hidden layer
        classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu', input_dim = 11))
        
        # Adding second layer 
        classifier.add(Dense(output_dim = 6, init ='uniform', activation = 'relu'))
        
        # Adding the output layer 
        classifier.add(Dense(output_dim = 1, init ='uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy']) 

        return classifier

classifier = KerasClassifier(build_fn = build_classifier) 
parameters = {'batch_size':[25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam', 'rmsprop', 'adagrad']
             }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring='accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parametes = grid_search.best_params_
best_accuracy = grid_search.best_score_








