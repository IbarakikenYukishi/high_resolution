#coding: UTF-8

#from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model,model_from_json
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,merge
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D,Conv2D,Conv2DTranspose
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from datetime import datetime as dt
import tensorflow as tf
from keras.backend import tensorflow_backend

import cv2
import numpy as np
import os
import h5py
import random

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


def load_model():
	model_path="../kasumi_models/model_default/model.json"
	if os.path.exists(model_path)==1:
		model = model_from_json(open('../kasumi_models/model_default/model.json').read())
		model.load_weights('../kasumi_models/model_default/weights.h5')
		model.summary()

		model.compile(loss='mse',
			optimizer=Adam(lr=LearningRate),
			metrics=['accuracy'])
		return model
	return 0

#モデルの準備
def model_generate():
	in_=Input(x_train.shape[1:])
	in_conv=Conv2D(nb_filters,(nb_conv,nb_conv),padding='same')(in_)
	act=Activation("relu")(in_conv)
#	batch_norm=BatchNormalization()(in_conv)

	for i in range(50):
		in_conv2=Conv2D(nb_filters,(nb_conv,nb_conv),padding='same')(act)
		batch_norm2=BatchNormalization()(in_conv2)
		act2=Activation("relu")(batch_norm2)
		in_conv_conv=Conv2D(nb_filters,(nb_conv,nb_conv),padding='same')(act2)
		batch_norm3=BatchNormalization()(in_conv_conv)
		merged=merge([batch_norm3,act],mode='sum')
		act=merged
	
	f = h5py.File("../kasumi_models/model_predict/weights.h5")
	layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
	weight_value_tuples = []
	print(layer_names)
#	print(len(model.layers))
	act3=Activation("relu")(act)
	deconv=Conv2DTranspose(3,(4,4),strides=(4,4),padding='same')(act3)
	model=Model(input=in_,output=deconv)
	model.summary()

	for k, name in enumerate(layer_names):
		if k >= len(model.layers)-7:
			# 全結合層の重みは読み込まない
			break
		g = f[name]
		weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
		if len(weight_names):
			weight_values = [g[weight_name] for weight_name in weight_names]
			layer = model.layers[k]
			symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
			if len(weight_values) != len(symbolic_weights):
				raise Exception('Layer #' + str(k) +
								' (named "' + layer.name +
								'" in the current model) was found to '
								'correspond to layer ' + name +
								' in the save file. '
								'However the new layer ' + layer.name +
								' expects ' + str(len(symbolic_weights)) +
								' weights, but the saved weights have ' +
								str(len(weight_values)) +
								' elements.')
			weight_value_tuples += zip(symbolic_weights, weight_values)
	K.batch_set_value(weight_value_tuples)
	f.close()





	model.compile(loss='mse',
		optimizer=Adam(lr=LearningRate),
		metrics=['accuracy'])
	return model

if __name__ == '__main__':
	config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
	session = tf.Session(config=config)
	tensorflow_backend.set_session(session)



	#各層のパラメータ
	nb_filters=128
	nb_conv=4
	nb_pool=2
	nb_epoch=500
	batch_sample=32
	image_w=32
	image_h=32
	split=4
	pixels=image_w*image_h*3
	nb_iter=int(300/batch_sample)
	LearningRate=0.001

	#モデルを保存するフォルダ名
	tdatetime=dt.now()
	tstr=tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr="../kasumi_models/model_"+tstr+"/"
	os.makedirs(tstr)
	train_dir="../kasumi_datasets/train"
	test_dir="../kasumi_datasets/test"

	#学習用データとテスト用データ
	x_train=[]
	y_train=[]
	x_test=[]
	y_test=[]
	x_train,y_train=load_data(train_dir,batch_sample)
#	print(x_train.shape[1:])
#	model=load_model()
#	if model==0:
#		model=model_generate()
	model=model_generate()

	for i in range(nb_epoch):
		if i%100==0:
			LearningRate*=0.8
			K.set_value(model.optimizer.lr,LearningRate)


		for j in range(nb_iter):
			x_train,y_train=load_data(train_dir,batch_sample)
			model.fit(x_train,y_train,batch_size=batch_sample,epochs=1,verbose=1,validation_split=0.1)

#			x_test,y_test=load_data(test_dir,batch_sample)
#			score=model.evaluate(x_test,y_test,verbose=0)
#			print('Test loss:',score[0])
#			print('Test accuracy:',score[1])

		if i%40==0:
			save_model(model,'model_'+str(i)+'.json','weights_'+str(i)+'.h5')

	save_model(model,'model_final.json','weights_final.h5')
