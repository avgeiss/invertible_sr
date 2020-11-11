#alternate plotter for fig 3 in CVPR paper

import numpy as np
import matplotlib.pyplot as plt

#plots training performance for all of the neural networkes
models = ['srcnn','dun','lapsrn','dbpn','edrn','dnsr','rdnsr']
panels = ['a','b','c','d','e','f','g','h']
models_long = ['SR-CNN','Dense U-Net','Laplacian Pyramid Net','Deep Back Projection','EDRN','Dense Net','Residual Dense Net']
#plotter.py
plt.figure(figsize=(18,8.5))
for i in range(len(models)):
    m = models[i]
    plt.subplot(2,4,i+1)
    loss1 = np.load('./models/' + m + '_tanh_psnr.npy')
    plt.plot(loss1,color='k')
    loss2 = np.load('./models/' + m + '_strict_psnr.npy')
    plt.plot(loss2,color=[1.0,0.3,0.2])
    if i == 4:
        lossl2 = np.load('./models/' + m + '_l2reg_psnr.npy')
        plt.plot(lossl2,color='b')
    plt.ylim([29.2,30.8])
    plt.xlim([1,300])
    plt.grid(True)
    plt.title(panels[i] + ') ' +  models_long[i])
    if i == 4:
        plt.legend(['Standard','Downsampling Aware','L2 Regularization'],loc='lower right')
        plt.plot(6,np.max(lossl2),color='b',marker='.')
    if i == 0 or i==4:
        plt.ylabel('Validation PSNR')
    if i>3:
        plt.xlabel('Training Epoch')
    plt.plot(6,np.max(loss1),color='k',marker='.')
    plt.plot(6,np.max(loss2),color=[1.0,0.3,0.2],marker='.')
    plt.yticks(np.arange(29.2,30.8,0.2))
   
#plot training performance for the bicubic downsampling case
ax = plt.subplot(2,4,8)
loss = np.load('./models/edrn_tanh_bicubic_psnr.npy')
mxl = np.max(loss)
plt.plot(loss,'k-')
loss = np.load('./models/edrn_strict_bicubic_psnr.npy')
plt.plot(loss,'-',color=[0.5,1.0,0.5])
plt.legend(['Standard','2D-AVG-DA'],loc='lower right')
plt.plot(6,np.max(loss),color=[0.5,1.0,0.5],marker='.')
plt.plot(6,mxl,color='k',marker='.')
plt.ylim([29.5,31.2])
plt.xlim([1,300])
plt.title('h) EDRN (Bicubic Downsampling)')
plt.xlabel('Training Epoch')
plt.ylabel('Validation PSNR')
plt.grid('on')
ax.yaxis.tick_right()
plt.savefig('./figures/train_perf.png',bbox_inches='tight',pad_inches=0.02,dpi=300)