import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

# Initilizes the model parameters
def def_model_param():
    GESTURE_CATEGORIES = len(CATEGORY_MAP)
    base_model = Sequential()
    base_model.add(SqueezeNet(input_shape=(225, 225, 3), include_top=False))
    base_model.add(Dropout(0.5))
    base_model.add(Convolution2D(GESTURE_CATEGORIES, (1, 1), padding='valid'))
    base_model.add(Activation('relu'))
    base_model.add(GlobalAveragePooling2D())
    base_model.add(Activation('softmax'))

    return base_model

# This function returns the numeric equivalent of the category/class name    
def label_mapper(val):
    return CATEGORY_MAP[val]

# Input images folder name
training_img_folder = 'training_images'

# Assign numbers to each of the categories.
CATEGORY_MAP = {
    "up": 0,
    "down": 1,
    "mute": 2,
    "play": 3,
    "chrome": 4,
    "nothing": 5
}
    
# Loading the input training images from all the folders into 'input_data' variable
input_data = []
for sub_folder_name in os.listdir(training_img_folder):
    path = os.path.join(training_img_folder, sub_folder_name)
    for fileName in os.listdir(path):
        if fileName.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, fileName))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (225, 225))
            #'input_data' stores the input image array and its corresponding label or category name
            input_data.append([img, sub_folder_name])

# Zip function to separate the 'img_data'(input image) & 'labels' (output text labels) 
img_data, labels = zip(*input_data)

# Converting text labels to numeric value as per CATEGORY_MAP
# Eg:- ["up","up",down","mute",..] -> [0,0,1,2,..]
# Python 'map' function takes 2 arguments:
# 1. A function (label_mapper)
# 2. An iterable 
labels = list(map(label_mapper, labels))


# Converting numeric labels and performing one hot encoding on them
# Eg:- [0,0,1,2] -> [[1 0 0 0 0 0]
#                    [1 0 0 0 0 0]
#                    [0 1 0 0 0 0]
#                    [0 0 1 0 0 0]]
labels = np_utils.to_categorical(labels)

# define the model
model = def_model_param()
model.compile(
    #optimizer=Adam(lr=0.001)
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit model
model.fit(np.array(img_data), np.array(labels), epochs=15)

print("Training Completed")

# save the trained model parameters into a .h5 file
model.save("gesture-model05_20.h5")
