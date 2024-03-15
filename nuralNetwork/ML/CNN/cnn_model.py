
'''
1.make model from youtube tutorial -2


2.make a model with chat gpt

3.make model form youtube tutorial -1



4.evalute the accuracy

'''

#test model 1

import tensorflow as tf

model1 = tf.keras.Sequencial([
    #cnn layer
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    
    #layer 2
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layer.MaxPool2D(2,2),
    
    #dense layer
    tf.keras.layersFlatten(), #input layer
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid") #output layer  

])


model.compile(
    loss = "binery_crosstropy",
    optimizer =  RMSprop(1r=0.001),
    metrics = ['acuracy']
)








# cnn modeling

model2 = tf.keras.Sequential([
    #cnn layer
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    
    #layer 2
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    #dense layer
    tf.keras.layers.Flatten(), #input layer
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10,activation="sigmoid") #output layer  
])


model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model2.fit(
    train_images,
    train_labels,
    epochs=30,   #iteration over the model to predict correct
    validation_data=(val_images,val_labels)
)



# Evaluate the model
test_loss, test_accuracy = model2.evaluate(val_images, val_labels)
print("Test Accuracy:", test_accuracy)


#test the model
# model2Pred = cnn.predict(array_of_image)

# model2Pred = np.argmax(model2Pred) #provide the class value

