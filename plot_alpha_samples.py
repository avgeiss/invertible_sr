# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:04:15 2020

@author: andrew
"""


from keras.models import load_model, Model
import neural_networks as nets
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

N = 192
eval_models = False
if eval_models:
    chips = np.zeros((5,N,N,3),dtype='uint8')
    sample_input = np.uint8(plt.imread('./data/bsds100/test/210088.png')*255)[np.newaxis,150:342,100:100+N,:]
    chips[4,:,:,:] = sample_input[0,:,:,:]
    #do the case without regularization:
    cnn = load_model('./models/edrn_strict_best')
    chips[1,:,:,:] = cnn.predict(sample_input).squeeze()
    cnn = Model(cnn.inputs[0],cnn.layers[-3].output)
    chips[0,:,:,:] = np.uint8(cnn.predict(sample_input).squeeze()*127.5+127.5)
    cnn = load_model('./models/edrn_alpha_10_best')
    chips[3,:,:,:]= cnn.predict(sample_input)[0].squeeze()
    cnn = Model(cnn.inputs[0],cnn.layers[-4].output)
    chips[2,:,:,:] = np.uint8(cnn.predict(sample_input).squeeze()*127.5+127.5)
    np.save('fish_alpha_samples.npy',chips)

chips = np.load('./fish_alpha_samples.npy')
img = np.ones((N*2,N*3,3),dtype='uint8')*255
for i in range(4):
    img[(i//2)*N:(i//2+1)*N,(i%2)*N:(i%2+1)*N,:] = chips[i,:,:,:]
img[N:N*2,N*2:N*3,:] = chips[4,:,:,:]

l = 24
labels = ['a','b','c','d','e']
def make_label(i):
    panel_label = Image.new('RGB', (l,l), color = (255,255,255))
    fnt = ImageFont.truetype('arial.ttf',16)
    d = ImageDraw.Draw(panel_label)
    d.text((2,2),'(' + labels[i] + ')', font=fnt, fill=(0, 0, 0))
    img = np.array(panel_label)
    img[:,0] = 0
    img[0,:] = 0
    img[-1,:] = 0
    img[:,-1] = 0
    return img
       
for i in range(4):
    x, y = (i//2)*N, (i%2)*N
    img[x:x+l,y:y+l,:] = make_label(i)
img[N:N+l,N*2:N*2+l] = make_label(4)

plt.imsave('./figures/alpha_samples_cvpr.png',img)
