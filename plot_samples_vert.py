# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:53:27 2020

@author: andrew
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import sys
# sys.path.append('C:\Windows\Fonts')
dset_names = ['SET5','SET14','BSDS100','Manga109','Urban100','Div2k']
net_names = ['Truth','EDRN','EDRN (w/ DA)']
hr = np.load('./sample_outputs.npy')
hr = hr[[0,7,8],:,:,:]
BW = 4
LW = 24
IW = 192
im = np.ones((LW+BW*5+IW*6,LW+BW*2+IW*3,3),dtype='uint8')*255

def make_label(text):
    lbl = Image.new('RGB', (IW,LW), color = (255,255,255))
    fnt = ImageFont.truetype('arial.ttf',16)
    d = ImageDraw.Draw(lbl)
    d.text((20,2), text, font=fnt, fill=(0, 0, 0))
    return np.array(lbl)

#tile images:
for i in range(3):
    x = LW+(IW+BW)*i
    im[0:LW,x:x+IW,:] = make_label(net_names[i])
    for j in range(6):
        y = LW+(IW+BW)*j
        im[y:y+IW,x:x+IW,:] = hr[i,j,:,:,:]
for j in range(6):
    y = LW+(IW+BW)*j
    im[y:y+IW,0:LW,:] = np.rot90(make_label(dset_names[j]))
    
        
plt.imsave('./figures/sample_outputs_vert.png',im)