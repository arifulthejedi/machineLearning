
'''
This model is based on convolution nural network(CNN)

this model can predict image classs which is trained


'''


import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf



'''
Data normalization

'''

def show_image_from_array(image_array, cmap='gray'):
    plt.imshow(image_array, cmap=cmap)
    plt.axis('off')



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



# Example usage
directory_path = "trainData"
images, labels = load_images_and_labels(directory_path,(100,100))

#label inversion in to integer array
labels = np.array(labels)
labels[labels == "healthy"] = 0
labels[labels == "infected"] = 1
labels = [int(num_str) for num_str in labels]


#image reshaping
images = np.array(images)/255



#cnn mideling1

# model2 = tf.keras.Sequential([
#     #cnn layer
#     tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation = 'relu',input_shape=(150,150,4)),
#     tf.keras.layers.MaxPool2D(2,2),
    
#     #layer 2
#     tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation = 'relu'),
#     tf.keras.layers.MaxPool2D(2,2),
    
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

#     #dense layer
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64,activation="relu"),
#     tf.keras.layers.Dense(10,activation="softmax") #output layer  
# ])




#cnn modeling 2
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


#model2.summary()

train_dataset = tf.data.Dataset.from_tensor_slices((images,labels)).batch(32)


# #train the model
model2.fit(
    train_dataset,
    epochs=30,   #iteration over the model to predict correct
    validation_data = train_dataset
)


# Evaluate the model
test_loss, test_accuracy = model2.evaluate(train_dataset)
print("Test Accuracy:", test_accuracy)





'''
Model prediction

'''

#load image
image_path = "testImg2.png"
image = Image.open(image_path)
image = image.resize((100, 100))  # Resize the image to match the input shape of the model
image = np.array(image) / 255.0  # Normalize pixel values

# Add an extra dimension to match the input shape expected by the model
image = np.expand_dims(image, axis=0)  # Shape: (1, 100, 100, 4)


predict = model2.predict(image)
predicted_class = np.argmax(predict)

print("classes: 0 = healthy | 1 = infected")
print("prediction:",predicted_class)