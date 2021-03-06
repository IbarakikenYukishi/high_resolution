#coding: UTF-8

#from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D,Conv2D,Conv2DTranspose
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from datetime import datetime as dt

import cv2
import numpy as np
import os

import random



def img2arr(img_path):
	image=cv2.imread(img_path)
	image=np.array(image)
	return image

def arr2img(arr):
	arr*=255
	cv2.imwrite("gray_scale.png", arr)

#モデルの保存
def save_model(model,name_model,name_weights):
	model_json_str=model.to_json()
	open(tstr+name_model,'w').write(model_json_str)
	model.save_weights(tstr+name_weights)
	print(name_model)
	print(name_weights)
	print("saved")

#評価や学習用データ生成
def load_data(directory,batch_size):
	caltech_dir=directory
	file_list=os.listdir(caltech_dir+"/input")
	file_num=len(file_list)

	batch_list=random.sample(range(file_num),batch_size)

	X=[]#インプット
	Y=[]#教師

	for batch_path in batch_list:
		img_input=cv2.imread(caltech_dir+"/input/"+str(batch_path)+".png")
		img_output=cv2.imread(caltech_dir+"/output/"+str(batch_path)+".png")

		img_input=np.array(img_input)
		img_output=np.array(img_output)

		X.append(img_input)
		Y.append(img_output)


	X=np.array(X)
	Y=np.array(Y)
	X=X.astype('float32')
	Y=Y.astype('float32')
	X/=255
	Y/=255

	print(X.shape)
	print(Y.shape)
	return X,Y

#モデルの準備
def model_generate():
	model=Sequential()
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same',input_shape=x_train.shape[1:]))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2DTranspose(3,(nb_conv,nb_conv),strides=(4,4),padding='same'))
	model.summary()

	model.compile(loss='mse',
		optimizer=Adam(lr=0.001),
		metrics=['accuracy'])
	return model

if __name__ == '__main__':
	#各層のパラメータ
	nb_filters=128
	nb_conv=4
	nb_pool=2
	nb_epoch=500
	batch_sample=128
	image_w=32
	image_h=32
	split=4
	pixels=image_w*image_h*3
	nb_iter=int(10000/batch_sample)

	#モデルを保存するフォルダ名
	tdatetime=dt.now()
	tstr=tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr="../models/model_"+tstr+"/"
	os.makedirs(tstr)
	train_dir="../datasets/train"
	test_dir="../datasets/test"

	#学習用データとテスト用データ
	x_train=[]
	y_train=[]
	x_test=[]
	y_test=[]
	x_train,y_train=load_data(train_dir,batch_sample)
	print(x_train.shape[1:])
	model=model_generate()

	for i in range(nb_epoch):
#		if i%100==0:
#			lr=model.optimizer.lr
#			K.set_value(model.optimizer.lr,lr*0.8)

		for j in range(nb_iter):
			x_train,y_train=load_data(train_dir,batch_sample)
			model.fit(x_train,y_train,batch_size=batch_sample,epochs=1,verbose=1,validation_split=0.1)

			x_test,y_test=load_data(test_dir,batch_sample)
			score=model.evaluate(x_test,y_test,verbose=0)
			print('Test loss:',score[0])
			print('Test accuracy:',score[1])

		save_model(model,'model_'+str(i)+'.json','weights_'+str(i)+'.h5')

	save_model(model,'model_final.json','weights_final.h5')