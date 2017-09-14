#coding: UTF-8


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D,Conv2D,Conv2DTranspose
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import os

import random
import shutil

caltech_dir="../kasumi_datasets/kasumi_output"
dir_list=os.listdir(caltech_dir)
file_list=[]


for dir_path in dir_list:
	abs_name=caltech_dir+'/'+dir_path;
	file_list.append(abs_name)

#	img_list=os.listdir(abs_name)
#	for img_path in img_list:
#		file_list.append(abs_name+'/'+img_path)

random.shuffle(file_list)

train_data=file_list[0:int(len(file_list)*0.9)]
test_data=file_list[int(len(file_list)*0.9):len(file_list)]

i=0
for train_path in train_data:
	shutil.copyfile(train_path,"../kasumi_datasets/train/"+str(i)+".png")
	i+=1

i=0
for test_path in test_data:
	shutil.copyfile(test_path,"../kasumi_datasets/test/"+str(i)+".png")
	i+=1
