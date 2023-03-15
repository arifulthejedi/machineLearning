# -*- coding: utf-8 -*-
"""
tenserFlow
"""


from sklearn.manifold import TSNE
from keras.datasets import mnist
"""
CSE483

Lab-final

Name:Ariful Islam

Id:202014011

"""



from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd  



#iris = load_iris()
#x = iris.data
#y = iris.target 




#load data
data=pd.read_csv('wine.data',delimiter = r',',header=None)
arr = data.to_numpy()

#number_of_features=arr.shape[1];


#separeting data and labe from data set
x = arr[:,1:13];

y = arr[:,0];


tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(x) 

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 3),
                data=df).set(title="Iris data T-SNE projection") 