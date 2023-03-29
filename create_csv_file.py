import csv
import pandas as pd
from sklearn.datasets import make_blobs



def create_csv(name,header,path):
    # Open the file for writing with a 'w' mode
    with open(path+name, mode='w', newline='') as csv_file:
      # Create a CSV writer object
      csv_writer = csv.writer(csv_file)
    
      # Write the data to the CSV file
      for row in [header]:
         csv_writer.writerow(row)




def insert_data(data,path):
   with open(path, mode='a', newline='') as csv_file:
      # Create a CSV writer object
      csv_writer = csv.writer(csv_file)

      # Write the new data to the CSV file
      for row in data:
         csv_writer.writerow(row)



#data making
def create(rows,cluster,std,features,integer = False):
   X, y = make_blobs(n_samples=rows, centers=cluster,cluster_std=std, random_state=42, n_features=features)
   if integer == True:
       X = X.astype(int)
   return [X,y]


#insert a column
def insert_column(path,name,data):
      df = pd.read_csv(path)

      #inserting the data to dataframe
      df[name] = data

      # Write the DataFrame back to the CSV file
      df.to_csv(path, index=False)

