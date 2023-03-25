import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import normalize, StandardScaler




#loading the data
X = pd.read_csv('/home/ariful/Desktop/python/ML/data/data.csv', header=0)


#droping the header
X = X.iloc[1:]


# Handling the missing values if any
X.fillna(method ='ffill', inplace = True)


#cleaning the data
X = np.array(X)

X = X[:,0:2]



##pre process the data##

# Scaling the data to bring all the attributes to a comparable level
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Normalizing the data so that the data
# approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)
 
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)
 




# Create linkage matrix using complete linkage
linkage_matrix = linkage(X_normalized, method='complete')



# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()



