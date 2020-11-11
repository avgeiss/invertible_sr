#traininer.py
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import neural_networks as nnets
import data_manager as dat
import numpy as np
import keras.backend as K

#set constants
BATCH_SIZE=16
EPOCH_SIZE = 1000*BATCH_SIZE
N_EPOCHS = 300
N_EPOCHS_LOW_LR = 100
HR = 192
network = 'edrn'
ds_type = 'bicubic'
postproc = 'tanh'
FNAME = './models/' + network + '_' + postproc + '_' + ds_type

#load the data:
if network == 'edrn':
    imhr = dat.load_images('div2k','train')      #change to 'train'
    imlr = dat.load_images('div2k','train_lr')
    test_hr = imhr[-10:]
    imhr = imhr[:-10]
    test_lr = imlr[-10:]
    imlr = imlr[:-10]

#preallocate training batches and prep test set:
inputs = np.zeros((EPOCH_SIZE,HR//4,HR//4,3),dtype='uint8')
targets = np.zeros((EPOCH_SIZE,HR,HR,3),dtype='uint8')

#build the neural network:
cnn = nnets.build_cnn(network,output_type=postproc,ds_type=ds_type)

#training loop:
psnr = []
mse = []
best_vloss = 100000.00
for epoch in range(N_EPOCHS):    
    #print epoch number, reduce LR if necessary:
    print(epoch)
    if epoch == N_EPOCHS-N_EPOCHS_LOW_LR:
        print('Reducing Learning Rate')
        K.set_value(cnn.optimizer.lr,0.1*K.eval(cnn.optimizer.lr))
    
    #train for an epoch:
    dat.bicub_batch(imlr,imhr,inputs,targets)
    cnn.fit(inputs,targets,batch_size=BATCH_SIZE)

    #evaluate test set
    cur_mse, cur_psnr, _ = dat.eval_test_set(cnn,test_hr,images_lr=test_lr)
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
