# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:53:27 2020

@author: andrew
"""

import data_manager as dm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import matplotlib.pyplot as plt
import numpy as np
import neural_networks as nets
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from skimage.io import imsave
eval_samples = True
import sys
sys.path.append('C:\Windows\Fonts')

if eval_samples:
    chip_ids = [[['./data/set5/test/bird.png'],12,80],
                [['./data/set14/test/baboon.png'],220,300],
                [['./data/bsds100/test/101087.png'],92,80],
                [['./data/manga109/test/TaiyouNiSmash.png'],952,300],
                [['./data/urban100/test/img_062.png'],280,300],
                [['./data/div2k/test/0882.png'],372,940]]
    
    NETWORK_NAMES = ['dun','lapsrn','dbpn','edrn','dnsr','rdnsr']
    
    chips = []
    for ci in chip_ids:
        im = dm.load_images('','',ci[0])[0]
        chips.append(im[ci[1]:ci[1]+192,ci[2]:ci[2]+192])
    
    chips
    
    #apply sr schemes to sample images:
    inputs = np.array(chips)
    hr = [inputs]
    for net in NETWORK_NAMES:
        print(net)
        cnn = load_model('./models/' + net + '_tanh_best')
        hr.append(list(cnn.predict(inputs,verbose=1)))
        cnn = load_model('./models/' + net + '_strict_best')
        hr.append(list(cnn.predict(inputs,verbose=1)))
        
    hr = np.array(hr)
    np.save('sample_outputs.npy',hr)


dset_names = ['SET5','SET14','BSDS100','Manga109','Urban100','Div2k']
net_names = [['Dense U-Net','LapSRN','DBPN'],['Enh. Res-Net','Dense Net','Res. Dense Net']]
for imnum in range(2):
    hr = np.load('./sample_outputs.npy')
    hr = np.uint8(np.round(hr))
    BS = 6
    vbuf = np.ones((BS,192*6+BS*7,3),dtype='uint8')*255
    hbuf = np.ones((192,BS,3),dtype='uint8')*255
    rows = [vbuf]
    if imnum == 0:
        imrng = [0,1,2,3,4,5,6]
    else:
        imrng = [0,7,8,9,10,11,12]
    for i in imrng:
        row = [hbuf]
        for j in range(6):
            row.append(hr[i,j,:,:,:])
            row.append(hbuf)
        row = np.concatenate(row,axis=1)
        rows.append(row)
        rows.append(vbuf)
    chipimg = np.concatenate(rows,axis=0)
    
    collbl = Image.new('RGB', (192*6+6*7,32), color = (255,255,255))
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    fnt = ImageFont.truetype('arial.ttf',16)
    d = ImageDraw.Draw(collbl)
    for i in range(6):
        d.text((i*(192+BS)+BS,10), dset_names[i], font=fnt, fill=(0, 0, 0))
    chipimg = np.concatenate([np.array(collbl),chipimg],axis=0)
    
    
    ylbls = ['Truth']
    for i in range(len(net_names[imnum])):
        ylbls.append(net_names[imnum][i])
        ylbls.append(net_names[imnum][i] + ' (DA)')
    ylbls.reverse()
    rowlbl = Image.new('RGB', (192*7+6*8+32,32), color = (255,255,255))
    fnt = ImageFont.truetype('arial.ttf',16)
    d = ImageDraw.Draw(rowlbl)
    for i in range(7):
        d.text((i*(192+BS)+BS+3,10), ylbls[i], font=fnt, fill=(0, 0, 0))
    rowlbl = np.flip(np.array(rowlbl).transpose((1,0,2)),axis=0)
    chipimg = np.concatenate([rowlbl,chipimg],axis=1)
    imsave('./figures/samples' + str(imnum) + '.png',chipimg)