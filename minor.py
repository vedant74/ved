# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input/plant-seedlings-classification/"))

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 33
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

root = '../input/plant-seedlings-classification/train'
folders = os.listdir(root)
X = []
Y = []
names={}
ptr = 0
for folder in  folders:
     names[ptr]=folder
     files = os.listdir(os.path.join(root,folder))
     for file in files:
            image_path = os.path.join(os.path.join(root,folder,file))
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_segmented = segment_plant(img)
            image_sharpen = sharpen_image(image_segmented)
            img = cv2.resize(image_sharpen,(128,128))
            img=img/255
            X.append(img)
            Y.append(ptr)
     ptr+=1

X = np.array(X)
Y = np.array(Y)
names    

def display_dataset(X,Y, h=128, w=128, rows=5, cols=2, display_labels=True):
    f, ax = plt.subplots(cols, rows)
    for i in range(rows):
        for j in range(cols):
            index=np.random.randint(0,X.shape[0])
            ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
            ax[j,i].set_title(Y[index])
    plt.xticks()
    plt.show()

display_dataset(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(128,128,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
opt = Adam(lr=0.00001)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

model = Sequential()
model.add(restnet)
model.add(Dense(256, activation='relu', input_dim=(128,128,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
Y_train = convert_to_one_hot(Y_train, 12)
Y_test = convert_to_one_hot(Y_test, 12)
Y_train=Y_train.T
Y_test=Y_test.T
Y_train.shape
Y_train.shape


history = model.fit(X_train,Y_train,batch_size=33,
                              epochs=120,
                              verbose=1,validation_data=(X_test, Y_test))

model.save("m1.h5")
