#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:32:01 2020

@author: andrew
"""

#traininer.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#variables to change for different training runs:##############################
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
resume=False
FNAME = './models/edrn_alpha_10'
##############################################################################

import neural_networks as nnets
import data_manager as dat
import numpy as np
import gc
import keras.backend as K
from keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')#make tensorflow less talkative

#set training constants:
EPOCH_SIZE = 16000
BATCH_SIZE = 16
N_EPOCHS = 300
N_EPOCHS_LOW_LR = 100
HR = 192

#get the neural network:
if resume:
    cnn = load_model('./models/edrn_alpha_10_last')
    psnr = list(np.load('./models/edrn_alpha_10_psnr.npy'))
    mse = list(np.load('./models/edrn_alpha_10_mse.npy'))
    best_vloss = np.min(mse)
else:
    cnn = nnets.build_cnn('edrn','alpha')
    psnr, mse, best_vloss = [], [], 10000000.0
    
#load the data:
images = dat.load_images('div2k','train')
test_set = images[-10:]
images = images[:-10]
batch = np.zeros((EPOCH_SIZE,HR,HR,3),dtype='uint8')

#training loop:
for epoch in range(N_EPOCHS):
    gc.collect()
    
    #print epoch number, reduce LR if necessary:
    print(epoch)
    if epoch == N_EPOCHS-N_EPOCHS_LOW_LR:
        print('Reducing Learning Rate')
        K.set_value(cnn.optimizer.lr,0.1*K.eval(cnn.optimizer.lr))
    
    #train for an epoch:
    dat.batch(images,batch)
    cnn.fit(batch,[batch,np.zeros((EPOCH_SIZE,))],batch_size=BATCH_SIZE)

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