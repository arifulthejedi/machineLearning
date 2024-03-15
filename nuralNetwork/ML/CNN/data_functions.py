'''

Data normalazation functions

image to array from directory

show image from an array

convert dictonery to csv

cdv to dictionery

'''

import os
import numpy as np
import PIL.Image as Image

'''
the function below convert all image found the directory
into array and label images based on the folder name inside the 
directory

'''


def load_images_and_labels(directory, target_size=(100, 100)):
    images = []
    labels = []

    # Iterate through each subdirectory (assuming each folder corresponds to a label)
    for label_name in os.listdir(directory):
        folder_path = os.path.join(directory, label_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through each image file in the folder
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                try:
                    # Open image using PIL
                    with Image.open(image_path) as img:
                        # Resize image
                        img = img.resize(target_size)
                        # Convert image to array
                        images.append(np.array(img))
                        labels.append(label_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return images, labels



# Example usage
directory_path = "trainData"
images, labels = load_images_and_labels(directory_path,(200,200))

#to rescale images 0 to 1 value
reshape= np.array(images)/255


'''
Reshape the data labeling
'''

labels = np.array(["healthy","infected","infected"])

labels[labels == "healthy"] = 0
labels[labels == "infected"] = 1

print(labels) #output [0,1,1]


'''
open image from array

'''
import matplotlib.pyplot as plt

def show_image_from_array(image_array, cmap='gray'):
    plt.imshow(image_array, cmap=cmap)
    plt.axis('off')
    plt.show()



'''
Convert dictonery to csv file

'''

import csv

def dict_of_arrays_to_csv(dictionary, file_path):
    """
    Write a dictionary of arrays to a CSV file.

    Parameters:
        dictionary (dict): The dictionary containing arrays to be written to the CSV file.
        file_path (str): The file path where the CSV file will be saved.
    """
    # Get the keys and determine the maximum length of arrays
    keys = list(dictionary.keys())
    max_length = max(len(array) for array in dictionary.values())
    
    # Write data to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(keys)
        
        # Write data rows
        for i in range(max_length):
            row_data = [dictionary[key][i] if i < len(dictionary[key]) else '' for key in keys]
            writer.writerow(row_data)

# Example usage:
example_dict = {'A': [[1,2,3],[3,4,5],[4,5,6]], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]}

# Save the dictionary to a CSV file
dict_of_arrays_to_csv(example_dict, 'hello.csv')





'''
Convert csv to dictionery

'''

import csv

def csv_to_dict(file_path):
    """
    Read a CSV file and convert it into a dictionary.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary where keys are the column headers and values are lists of column values.
    """
    data_dict = {}
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate over each row in the CSV file
        for row in reader:
            for key, value in row.items():
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(value)
    
    return data_dict

# Example usage:
csv_file_path = 'hello.csv'
data_dictionary = csv_to_dict(csv_file_path)




'''
Image color mode

'''

from PIL import Image

def detect_color_mode(image_path):
    try:
        img = Image.open(image_path)
        mode = img.mode
        if mode == "CMYK":
            print("Image is in CMYK color mode.")
        elif mode == "RGB":
            print("Image is in RGB color mode.")
        elif mode == "RGBA":
            print("Image is in RGBA color mode.")
        else:
            print("Image is in color mode:", mode)
    except FileNotFoundError:
        print("Error: File not found.")
    except PermissionError:
        print("Error: Permission denied to access the file.")
    except Exception as e:
        print("Error:", e)

# Provide the path to your image file
image_path = "trainData/healthy/1.png"  # Make sure your image is in PNG format
detect_color_mode(image_path)
