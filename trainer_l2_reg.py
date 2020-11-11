# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:59:39 2020

@author: andrew
"""

#traininer.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#variables to change for different training runs:##############################
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
CNN_NAME = 'edrn'
POSTPROC = 'l2reg'
resume = False
##############################################################################




import neural_networks as nnets
import data_manager as dat
import numpy as np
import gc
import keras.backend as K
import tensorflow as tf
from keras.models import load_model

tf.get_logger().setLevel('ERROR')#make tensorflow less talkative

#set training constants:
BATCH_SIZE = 16
EPOCH_SIZE = BATCH_SIZE*1000
N_EPOCHS = 300
N_EPOCHS_LOW_LR = 100
HR = 192
FNAME = './models/' + CNN_NAME + '_' + POSTPROC

#get the neural network:
cnn = nnets.build_cnn(CNN_NAME,POSTPROC)
if resume:
    cnn = load_model(FNAME + '_last')
cnn.summary()

#load the data:
images = dat.load_images('div2k','train')
test_set = images[-10:]
images = images[:-10]
batch = np.zeros((EPOCH_SIZE,HR,HR,3),dtype='uint8')
batch_lr = None

#training loop:
if resume:
    psnr = list(np.load(FNAME + '_psnr.npy'))
    mse = list(np.load(FNAME + '_mse.npy'))
    best_vloss = np.min(mse)
else:
    psnr = []
    mse = []
    best_vloss = 100000.00
    
for epoch in range(len(mse),N_EPOCHS):
    gc.collect()
    
    #print epoch number, reduce LR if necessary:
    print(epoch)
    if epoch == N_EPOCHS-N_EPOCHS_LOW_LR:
        print('Reducing Learning Rate')
        K.set_value(cnn.optimizer.lr,0.1*K.eval(cnn.optimizer.lr))
    
    #train for an epoch:
    dat.batch(images,batch)
    del batch_lr
    batch_lr = dat.pixelate(batch,scale=[4,4],axis=[1,2])
    cnn.fit(batch,[batch,batch_lr],batch_size=BATCH_SIZE)

    #evaluate test set
    cur_mse, cur_psnr, _ = dat.eval_test_set(cnn,test_set)
    print('MSE = ' + str(cur_mse) + '         PSNR = ' + str(cur_psnr))
    mse.append(cur_mse)
    psnr.append(cur_psnr)
    np.save(FNAME + '_mse.npy',mse)
    np.save(FNAME + '_psnr.npy',psnr)
    
    #save the model:
    cnn.save(FNAME + '_last')
    if cur_mse < best_vloss:
        best_vloss = cur_mse
        cnn.save(FNAME + '_best')