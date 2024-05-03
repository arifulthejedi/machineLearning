Arifitial Intilligence Model for Identifing Image Classification
=================================================================


Email: arifulthejedi@gmail.com 
Github: https://github.com/arifulthejedi

step-1:
install pythons and
os
numpy
PIL
matplotlib
tensorflow

these packages

step-2:
download and upzeap the CNN file


step-3:
go inside the CNN folder
and run  
python cnn_model

step-3:
include the file as the folwing description test the model and evaluate


Demo  Test 
==========
consider two class of image
The model should predict the class for given image

Data/train/healthy - contain all healthy leaf 
Data/train/infected - contain all infected leaf 
Data/test - contain all leaf photo for test for the established model

all train report shown the screen shot

there are three test performed which shown in screen shot




Functions in side the script
============================

def load_images_and_labels():
------------------------------
this function takes a folder path

return image array insde the folder 
and label array based on the name of folder

after get the label the label should convert into numaric value

Ex:
Data/train folder contain /infected and /healthy tow folder inside it all image placed in

load_images_and_labels("Data/train") this return image array and labels(["helthy","infected","healthy"....])

after call the function the labels shoul convert in to numaric 
Ex:
here healthy = 0 and infected = 1
["healthy","infected","healthy"] into [0,1,0]


mode2
------
this is the CNN model where the train and label shoul function
in the script there are mensioned


def CNN(path)
-----------
This function is for predict the given image data
the model predict inside the function

This function take a image path(must be .PNG formate)

return the predicted class in numaric form



use step
===========
1.include all pre classified image which is for train the model

 following structure of directory

 Data - this folder contain two folder /train and /test

 

 2.inside train folder all classified image should separate folder

 example 
  /infected - in this folder all infacted class image shoul in
  /healthy - in this folder all healthy class image should in

3. Note: the all image formate must be .PNG formate


In the folder there are some dummy data for demo purpose
