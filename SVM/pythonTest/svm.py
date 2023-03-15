# -*- coding: utf-8 -*-
"""
CSE483

Lab-final

Name:Ariful Islam

Id:202014011

"""


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


data=pd.read_csv('wine.data',delimiter = r',',header=None)
arr = data.to_numpy()

number_of_features=arr.shape[1];


#separeting data and labe from data set
data=arr[:,1:13];
label = arr[:,0];



#spliting the data 
x_train, x_test ,y_train,y_test = train_test_split(data,label,test_size=.4,random_state =110);



#Create a svm Classifier linear
clf = svm.SVC(kernel='linear') # Linear Kernel


sig = svm.SVC(kernel='sigmoid') # sigmoid Kernel

#Train the model using the training sets linear
clf.fit(x_train, y_train)

#Train the model using the training sets sigmoid
sig.fit(x_train, y_train)



#Predict the response for test dataset linear
y_pred_linear = clf.predict(x_test)

#Predict the response for test dataset gimoid
y_pred_sigmoid = sig.predict(x_test)





# Model Accuracy kernel linear
print("Accuracy-linear:",metrics.accuracy_score(y_test,y_pred_linear))


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision Linear:",metrics.precision_score(y_test, y_pred_linear, average=None))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall Linear:",metrics.recall_score(y_test, y_pred_linear,average=None))




# Model Accuracy sigmoid sigmoid
#print("Accuracy-sigmoid:",metrics.accuracy_score(y_test,y_pred_sigmoid))

# Model Precision
#print("Precision sigmoid:",metrics.precision_score(y_test, y_pred_sigmoid))

# Model Recall
#print("Recall sigmoid:",metrics.recall_score(y_test, y_pred_sigmoid))