import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score




#loading the data
X = pd.read_csv('/home/ariful/Desktop/python/ML/data/segmentation_data.csv', header=0)


# Handling the missing values if any
X.fillna(method ='ffill', inplace = True)

X = np.array(X)

##pre process the data##
y_true = X[:,7]

X = X[:,1:7]



# Building the OPTICS Clustering model
optics_model = OPTICS(min_samples = 5, xi = 0.02, metric='euclidean')

# Training the model
optics_model.fit(X)


 
# Storing the cluster labels of each points
labels = optics_model.labels_[optics_model.ordering_]

silhouette_score(X,labels)

accuracy_score(y_true,labels)