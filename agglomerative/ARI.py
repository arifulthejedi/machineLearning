import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score





#loading the data
X = pd.read_csv('/home/ariful/Desktop/python/ML/data/CarbonEmissionIndia.csv', header=0)



# Handling the missing values if any
X.fillna(method ='ffill', inplace = True)


#cleaning the data
X = np.array(X)

label = X[:,4]
X = X[:,1:4]


# Run agglomerative clustering
agg = AgglomerativeClustering(n_clusters=4)
y_pred = agg.fit_predict(X)

# Calculate ARI
ari = adjusted_rand_score(label,y_pred)
print("ARI:", ari)