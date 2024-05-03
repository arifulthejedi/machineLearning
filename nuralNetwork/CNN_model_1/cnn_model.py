
'''
This model is based on convolution nural network(CNN)

this model can predict image classs based on trained data


'''


import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf



'''
Data normalization functions

'''

#this function go to a folder fetch all image and convert into array and lavel
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



'''
Test the model for prediction

'''
#load image for apply to model for prediction
def CNN(path):
    #load image
    image_path = path
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize the image to match the input shape of the model
    image = np.array(image) / 255.0  # Normalize pixel values

    # Add an extra dimension to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)  # Shape: (1, 100, 100, 4)
    predict = model2.predict(image)
    predicted_class = np.argmax(predict)

    return predicted_class






#cnn modeling 
model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 4)), # Assuming input images are 100x100 pixels with 4 channels
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((images,labels)).batch(32)


#train the model
model2.fit(
    train_dataset,
    epochs=30,   #iteration over the model to predict correct
    validation_data = train_dataset
)


# Example usage
directory_path = "Data/train"
images, labels = load_images_and_labels(directory_path,(100,100))

#label inversion in to integer array
labels = np.array(labels)
labels[labels == "healthy"] = 0
labels[labels == "infected"] = 1
labels = [int(num_str) for num_str in labels]


#image reshaping
images = np.array(images)/255




'''
Use the model for prediction
here the lebel should be differet 
so first print labael with numeric value
'''

print("classes: 0 = healthy | 1 = infected")
print("prediction:",CNN("Data/test/testImg6.PNG"))




