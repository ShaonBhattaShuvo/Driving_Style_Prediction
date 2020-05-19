# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:54:55 2020

@author: Shaon Bhatta Shuvo
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
driver = pd.read_csv('Training_Data/Train.csv')
weather = pd.read_csv('Training_Data/Train_WeatherData.csv')
travelling = pd.read_csv('Training_Data/Train_Vehicletravellingdata.csv')
data_temp = pd.merge(travelling,weather)
data = pd.merge(data_temp,driver)
print(data)
data.drop(["ID","V7","V14"], axis = 1, inplace = True) 

#checking null values
#print(data.head())
print(data.isnull().values.sum()) #total null values
print(data.isnull().sum()) #column wise null values

#taking care of null/mining values
data["V11"].fillna(data["V11"].mean(), inplace = True) 
data["V12"].fillna(data["V12"].mean(), inplace = True) 
data["V15"].fillna(data["V15"].mean(), inplace = True) 
data["V16"].fillna(data["V16"].mean(), inplace = True) 
data["V17"].fillna(data["V17"].mean(), inplace = True) 
print(data.isnull().sum())

#Shuffling the dataframe 
#data = data.reindex(np.random.permutation(data.index))

#seperating input and target valuDrivingStylees
inputs_all = data.copy()
inputs_all.drop(["V1","DrivingStyle"], axis=1, inplace =True)
targets_all = data['DrivingStyle']
#converting categorical data to numerical values using one hot encoding
from sklearn import preprocessing
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(data[['V18','V19','V13']]).toarray())
#adding new encoded columns and removing old values from the datasetinputs_all = unscaled_inputs_all.join(encoder_df)
inputs_all = inputs_all.join(encoder_df)

inputs_all.drop(["V18","V19","V13"], axis = 1, inplace = True)

#Balancing the dataset to obtain accurate classification result

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
#counting instaces of each target class
#targets_all = data.iloc[:,-1].values
class_1 = 0
class_2 = 0
class_3 = 0
count_else = 0
for i in range(targets_all.shape[0]):
    if targets_all[i] == 1:
        class_1 += 1
    elif targets_all[i] == 2:
        class_2 += 1
    elif targets_all[i] == 3:
        class_3 +=1
    else:
        count_else +=1
print("Class 1: ",class_1)
print("Class 2: ",class_2)
print("Class 3: ",class_3)
print("Ohters : ",count_else)

#initializing count variable to check how many of them are obove class 1
class2_targets_counter = 0
class3_targets_counter = 0

# Declare a variable that will do that:
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 2:
        class2_targets_counter += 1
        if class2_targets_counter > class_1:
            indices_to_remove.append(i)
    elif targets_all[i] == 3:
        class3_targets_counter += 1
        if class3_targets_counter > class_1:
            indices_to_remove.append(i)
# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.
inputs_all.drop(indices_to_remove, axis=0, inplace =True)
targets_all.drop(indices_to_remove, axis=0, inplace =True)
#Resetting indices from 0 to fill the gaps 
inputs_all.reset_index(drop=True, inplace=True)
targets_all.reset_index(drop=True, inplace=True)
#counting instaces of each target class
targets_all.value_counts()

#Scalling the data
# We will use the sklearn preprocessing library, as it will be easier to standardize the data.
from sklearn import preprocessing
inputs_all = preprocessing.scale(inputs_all)

# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.
# Since we will be batching, we want the data to be as randomly spread out as possible
shuffled_indices = np.arange(inputs_all.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
inputs_all = inputs_all[shuffled_indices]
targets_all = targets_all[shuffled_indices]

#split the dataset into train validation and testset
# Count the total number of samples
samples_count = inputs_all.shape[0]
# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count
# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = inputs_all[:train_samples_count]
train_targets = targets_all[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = inputs_all[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = targets_all[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = inputs_all[train_samples_count+validation_samples_count:]
test_targets = targets_all[train_samples_count+validation_samples_count:]

#counting instaces of each target class to check if the dirstribution is iid or not
train_targets.value_counts()
validation_targets.value_counts()
test_targets.value_counts()

# Save the three datasets in *.npz.
#it is extremely valuable to name them in such a coherent way!
np.savez('driving_data_train', inputs=train_inputs, targets=train_targets)
np.savez('driving_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('driving_data_test', inputs=test_inputs, targets=test_targets)

#Loading npz (training, validation and test files)

npz = np.load('driving_data_train.npz')
X_train = npz['inputs'].astype(np.float)
y_train = npz['targets'].astype(np.int)
npz = np.load('driving_data_validation.npz')
X_validation = npz['inputs'].astype(np.float)
y_validation = npz['targets'].astype(np.int)
npz = np.load('driving_data_test.npz')
X_test = npz['inputs'].astype(np.float)
y_test = npz['targets'].astype(np.int)

#Creating deep learning model to train the data
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

input_size = 23
output_size = 4
hidden_layer_size = 200
early_stopping = tf.keras.callbacks.EarlyStopping(patience =10)
model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
batch_size =100
max_epochs =150
model.fit(X_train,y_train,batch_size=batch_size,epochs=max_epochs,
          validation_data=(X_validation,y_validation),
          callbacks = [early_stopping],
          verbose=2)

#Evaluating model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%d'.format(test_loss, test_accuracy*100)) 

#Decision Tree
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test,y_pred)
print('Accuracy = {0:.2f}%'.format(accuracy*100))
print( "{0}".format(metrics.classification_report(y_test,y_pred)))

#Plotting the tree
tree.plot_tree(classifier)
#Plotting the tree in pdf
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("DriverBehaviorTree", view= True)
