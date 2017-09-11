#coding: UTF-8


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from datetime import datetime as dt

import cv2
import numpy as np
import os

import random

batch_sample=64

image_input_w=8
image_input_h=8
image_output_w=32
image_output_h=32
pixels=image_w*image_h*3

def generate_data(arr,file_list):
	arr*=255
	arr=arr.astype('uint8')

	for i in range(len(file_list)):
		image=arr[i,:,:,:]
		cv2.imwrite(file_list[i],image)				


def load_data(directory):
	caltech_dir=directory
	file_list=os.listdir(caltech_dir)
	file_num=len(file_list)
	print(file_list)

	X=[]#インプット
	filename_list=[]

	for file_path in file_list:
		from_path=caltech_dir+"/"+str(file_path)
		to_path=tstr
		img_input=cv2.imread(from_path)

		image_b=cv2.resize(img_input,(image_output_h,image_output_w),interpolation=cv2.INTER_CUBIC)
		head,tail=os.path.splitext(file_path)
		cv2.imwrite(to_path+head+'_bicubic.png',image_b)
#		cv2.imwrite("output_bicubic.png",image_b)
		
		img_input=np.array(img_input)
		X.append(img_input)
		filename_list.append(to_path+head+'_output.png')


	X=np.array(X)
	X=X.astype('float32')
	X/=255
	filename_list=np.array(filename_list)

	print(X.shape)
	return X,filename_list

if __name__=='__main__':

	tdatetime=dt.now()
	tstr=tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr="../datasets/predict/predict_"+tstr+"/"
	os.makedirs(tstr)

	# 学習結果を読み込む
	model = model_from_json(open('../models/model_predict/model.json').read())
	model.load_weights('../models/model_predict/weights.h5')
	model.summary();
	model.compile(loss='mse',
		optimizer=Adam(),
		metrics=['accuracy'])
	input_data,file_list=load_data("../datasets/predict/data/")
	arr=model.predict(input_data)
	generate_data(arr,file_list)
