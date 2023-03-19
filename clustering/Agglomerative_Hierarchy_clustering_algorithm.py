import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
import os

data_dir = '/home/ariful/Desktop/python/ML/clustering'

os.chdir(data_dir)

X = df = pd.read_csv('qsar_fish_toxicity.csv',sep=';',header=None)

# Handling the missing values if any
X.fillna(method ='ffill', inplace = True)
 
# Scaling the data to bring all the attributes to a comparable level
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Normalizing the data so that the data
# approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)
 
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

#set the columns of raw unscaled data
X_normalized.columns = X.columns



# Compute the linkage matrix
Z = linkage(X, 'ward')

# Plot the dendrogram
fig, ax = plt.subplots(figsize=(100, 60))
dendrogram(Z)
plt.show()

