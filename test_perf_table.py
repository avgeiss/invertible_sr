#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:06:18 2020

@author: andrew
"""

#creates table 1 from the paper

import data_manager as dm
import neural_networks as nets
from keras.models import load_model
import numpy as np

#needed to re-compute these values for lapsrn and dbpn only
dsets = ['set5','set14','bsds100','manga109','urban100','div2k']
cnns = ['srcnn','lapsrn','dun','dbpn','dnsr','edrn','rdnsr']

PSNR = []
SSIM = []
for dset in dsets:
    images = dm.load_images(dset,'test')
    if dset == 'set14':
        images[2] = np.tile(images[2][:,:,np.newaxis],(1,1,3))
    psnr_col = []
    ssim_col = []
    for cnn_name in cnns:
        print('Applying ' + cnn_name + ' to ' + dset + '...')
        cnn = load_model('./models/' + cnn_name + '_tanh_best')
        _,psnr,ssim = dm.eval_test_set(cnn,images,compute_ssim=True)
        psnr_col.append(psnr)
        ssim_col.append(ssim)
        print(str(psnr) + '     ' + str(ssim))
        cnn = load_model('./models/' + cnn_name + '_strict_best')
        _,psnr,ssim = dm.eval_test_set(cnn,images,compute_ssim=True)
        print(str(psnr) + '     ' + str(ssim))
        psnr_col.append(psnr)
        ssim_col.append(ssim)
    PSNR.append(psnr_col)
    SSIM.append(ssim_col)

print('Generating Table...')
PSNR = np.array(PSNR).T
SSIM = np.array(SSIM).T
txtfile = open('./figures/perf_table.txt','w')
def addvals(s1,s2,n1,n2,form):
    if n1>n2:
        s1 += '\\textbf{'
    elif n2>n1:
        s2 += '\\textbf{'
    s1 += format(n1,form).lstrip('0')
    s2 += format(n2,form).lstrip('0')
    if n1>n2:
        s1 += '}'
    elif n2>n1:
        s2 += '}'
    return s1,s2
for n in range(len(cnns)):
    tanh = ''
    strict = ''
    for d in range(len(dsets)):
        tanh += ' & '
        strict += ' & '
        tanh,strict=addvals(tanh,strict,PSNR[n*2,d],PSNR[n*2+1,d],'.2f')
        tanh += '/'
        strict += '/'
        tanh,strict=addvals(tanh,strict,SSIM[n*2,d],SSIM[n*2+1,d],'.4f')
    tanh += ' \\\\\n'
    strict += ' \\\\\n'
    txtfile.write(tanh)
    txtfile.write(strict)
    txtfile.write('\hline \n')
txtfile.close()