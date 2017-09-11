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

#各層のパラメータ
nb_filters=128
nb_conv=3
nb_pool=2
nb_classes=10
nb_epoch=200

batch_sample=64#100
image_w=32
image_h=32
split=4
pixels=image_w*image_h*3
nb_iter=int(10000/batch_sample)

def img2arr(img_path):
	image=cv2.imread(img_path)
	image=np.array(image)
	return image

def arr2img(arr):
	arr*=255
	cv2.imwrite("gray_scale.png", arr)

def img_trim(image,save_dir):
	image=image[62-2:62*3+2,62-2:62*3+2]
	head,tail=os.path.splitext(save_dir)
	cv2.imwrite(head+'.png',image)

def img_comp(image,save_dir):
	compress=np.zeros([8,8,3])
	for i in range(int(image_w/4)):
		for j in range(int(image_h/4)):
			temp=image[i*4:(i+1)*4,j*4:(j+1)*4]
			temp=temp.astype('float32')
			temp=temp.sum(axis=1)
#			print("temp_raw")
#			print(temp)
			temp=temp.sum(axis=0)
			temp/=16
			temp=temp.astype('uint8')
			compress[i,j]=temp
	cv2.imwrite(save_dir,compress)
#	return compress




def load_data(directory,batch_size):
	caltech_dir=directory
	file_list=os.listdir(caltech_dir)
	x_train=[]#学習用データのインプット
	y_train=[]#学習用データの教師
	file_list=random.sample(file_list,batch_size)

	for file_path in file_list:
		image=cv2.imread(caltech_dir+"/"+file_path)
		image=image[1:image_h+1,1:image_w+1]
		image_train=cv2.resize(image,(int(image_h/split),int(image_w/split)))
		image=np.array(image)
		image_train=np.array(image_train)

		y_train.append(image)
		x_train.append(image_train)


	x_train=np.array(x_train)
	y_train=np.array(y_train)
	x_train=x_train.astype('float32')
	y_train=y_train.astype('float32')
	x_train/=255
	y_train/=255

	print(x_train.shape)
	print(y_train.shape)
	return x_train,y_train

'''
file_list=os.listdir("./lfw_eval")
for file_path in file_list:
	image=cv2.imread("./lfw_eval/"+file_path)
	img_trim(image,"./lfw_eval_trim/"+file_path)
'''

comp_dir="./lfw_eval"
file_list=os.listdir(comp_dir)
for file_path in file_list:
	image=cv2.imread(comp_dir+"/"+file_path)
	img_comp(image,comp_dir+"_comp/"+file_path)
