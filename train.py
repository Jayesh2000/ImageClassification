# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#print(os.listdir("../input/gnr638_train/train/aircrafts"))

filenames = ['aircrafts','ships','none']
categories = []
v1 = "../input/gnr638_train/train/"
print(filenames)
arr = []
for idx,imageId in enumerate(filenames):
#     for idx,folder in enumerate(imageId):    
    files = os.listdir(v1 + imageId)
    for x in files:
        l1 = [0]*4
        l1[0] = v1+imageId+"/"+x
        l1[idx+1] = 1
        arr.append(l1)
    
print(arr)

#     print(imageId)
#     label = imageId.split('/')[0]
#     if label == 'aircrafts':
#         categories.append(1)
#     if label == 'ships':
#         categories.append(2)    
#     else:
#         categories.append(3)

# # df = pd.DataFrame({
# #     'imageId': filenames,
# #     'label': categories
# # })
# # df.head()
arr = np.array(arr)
print(arr)


import scipy
np.random.shuffle(arr)
images = arr[:,0]
labels = arr[:,1:]

image_size = 200
imagearr = []
for img_path in images:
    # img = Image.open(img_path)
    # img = img.resize((image_size, image_size))
    img= scipy.misc.imread(img_path)
    img=scipy.misc.imresize(img,(image_size,image_size,3),interp="cubic")
    # print(img.shape)
    # arr = []
    imagearr.append(img)

# print(len(imagearr))
imagearr = np.array(imagearr)
# trainimg = imagearr[:1300]
# trainlbl = labels[:1300]
# valimg = imagearr[1300:]
# vallbl = labels[1300:]

imagearr = imagearr.astype(np.float32)
# imagearr = imagearr/255.0

from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
import keras

image_size = 200
input_shape = (image_size, image_size, 3)

epochs = 5
batch_size = 64

# pre_trained_model = keras.applications.vgg19.VGG19(input_shape=input_shape, include_top=False, weights="imagenet")
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
pre_trained_model.summary()
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
# x = GlobalMaxPooling2D()(last_output)
x = Flatten()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(256, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()

trainimg=imagearr
trainlbl=labels

history = model.fit(trainimg, trainlbl, epochs=40, batch_size=200,validation_split=0)
model.summary()
model.save('saved_model.h5')

