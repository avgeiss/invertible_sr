from skimage.io import imread
from glob import glob
import numpy as np
from scipy.signal import convolve2d
    
def load_images(dset='div2k',tset='train'):
    #get a list of the image files:
    dir_name = './data/' + dset + '/' + tset + '/'
    files = [*glob(dir_name + '*.png'),
             *glob(dir_name + '*.jpg')]
    files.sort()
    
    #load the data:
    print('Loading Data...')
    images = []
    for f in files:
        im = imread(f)
        images.append(im)
    
    return images

def batch(images,batch_data):
    #batch(images,batch_data):
    #
    #fills 'batch_data' with randomly selected and augmented image chips from
    #'images'
    ES = batch_data.shape[0]    #size of epoch
    HR = batch_data.shape[1]    #size of hi-res image chips
    NI = len(images)            #number of images to sample from
    
    for i in range(ES):
        #get a random sample:
        im_idx = np.random.randint(0,NI)
        sz = images[im_idx].shape
        x = np.random.randint(0,sz[1]-HR)
        y = np.random.randint(0,sz[0]-HR)
        sample = images[im_idx][y:y+HR,x:x+HR,:]
        
        #do some augmentation:
        if np.random.choice([True,False]):
            sample = np.flip(sample,axis=0)
        if np.random.choice([True,False]):
            sample = np.flip(sample,axis=1)
        nrots = np.random.randint(0,4)
        sample = np.rot90(sample,nrots)
        
        #assign to epoch matrices:
        batch_data[i,:,:,:] = sample
        
def bicub_batch(im_lr,im_hr,inputs,targets):
    #bicub_batch(im_lr,im_hr,inputs,targets):
    #
    #fills 'inputs' with randomly selected image chips from 'im_lr' and fills
    #'targets' with the corresponding hi-res image chips from 'im_hr'
    
    ES = inputs.shape[0]    #size of epoch
    LR = inputs.shape[1]    #size of lo-res image chips
    NI = len(im_lr)            #number of images to sample from
    
    for i in range(ES):
        #get a random sample:
        im_idx = np.random.randint(0,NI)
        sz = im_lr[im_idx].shape
        x = np.random.randint(0,sz[1]-LR)
        y = np.random.randint(0,sz[0]-LR)
        sample_lr = im_lr[im_idx][y:y+LR,x:x+LR,:]
        sample_hr = im_hr[im_idx][4*y:4*y+4*LR,4*x:4*x+4*LR,:]
        
        #do some augmentation:
        if np.random.choice([True,False]):
            sample_lr = np.flip(sample_lr,axis=0)
            sample_hr = np.flip(sample_hr,axis=0)
        if np.random.choice([True,False]):
            sample_lr = np.flip(sample_lr,axis=1)
            sample_hr = np.flip(sample_hr,axis=1)
        nrots = np.random.randint(0,4)
        sample_lr = np.rot90(sample_lr,nrots)
        sample_hr = np.rot90(sample_hr,nrots)
        
        #assign to epoch matrices:
        inputs[i,:,:,:] = sample_lr
        targets[i,:,:,:] = sample_hr
        
def ssim(imx,imy,window=4):
    #computes structural similarity index
    
    imx = np.double(imx)/255.0
    imy = np.double(imy)/255.0
    
    #constants:
    c1 = 0.01**2
    c2 = 0.03**2
    c3 = c2/2.0
    
    #makes a gaussian filter
    def gauss(R):
        x, y = np.meshgrid(np.arange(-R,R),np.arange(-R,R))
        g = np.exp(-((x**2 + y**2)/(2.0*(2*R)**2)))
        return g/np.sum(g)
    filt = gauss(window)
    
    #patch statistics
    mux = convolve2d(imx,filt,mode='same',boundary='symm')
    muy = convolve2d(imy,filt,mode='same',boundary='symm')
    muxy = mux*muy
    mux2 = mux*mux
    muy2 = muy*muy
    sxy = convolve2d(imx*imy,filt,mode='same',boundary='symm')-muxy
    sx2 = convolve2d(imx*imx,filt,mode='same',boundary='symm')-mux2
    sy2 = convolve2d(imy*imy,filt,mode='same',boundary='symm')-muy2
    sxsy = np.sqrt(np.abs(sx2*sy2))
    
    #luminance, contrast and structure:
    l = (2*muxy+c1)/(mux2+muy2+c1)
    c = (2*sxsy + c2)/(sx2+sy2+c2)
    s = (sxy+c3)/(sxsy+c3)
    ssim = l*c*s
    
    return np.nanmean(ssim)

def eval_test_set(cnn,images,images_lr=None,compute_ssim=False,HR = 192,SF=4):
    #computes mse, psnr, and ssim for applying a cnn to a test set of images
    LR = HR//SF
    
    #converts RGB image to lumosity data
    def lumos(im):
        im = np.double(im)/255.0
        return (16 + 65.481*im[...,0] + 128.553*im[...,1] + 24.966*im[...,2])/255.0
    
    PSNR = []
    MSE = []
    SSIM = []
    for i in range(len(images)):
        inputs = []
        targets = []
        im = images[i]
        sz = im.shape
        im = im[:sz[0]-(sz[0]%SF),:sz[1]-(sz[1]%SF),:]
        
        #break each image into chips appropriately sized for the CNN:
        for j in range(0,im.shape[0]//SF-LR,LR//2):
            for k in range(0,im.shape[1]//4-LR,LR//2):
                targets.append(im[SF*j:SF*j+HR,SF*k:SF*k+HR,:])
                if images_lr is not None:
                    inputs.append(images_lr[i][j:j+LR,k:k+LR])
                else:
                    inputs.append(targets[-1])
                    
        #apply the CNN:
        inputs = np.array(inputs)
        targets = np.array(targets)
        outputs = cnn.predict(inputs)
        if type(outputs)==list:
            outputs = outputs[0]
        
        #only compute errors on center of each chip:
        targets = targets[:,LR:-LR,LR:-LR,:]
        outputs = outputs[:,LR:-LR,LR:-LR,:]
        
        #compute error:
        targets = np.double(targets)
        MSE.append(np.mean((targets-outputs)**2))
        y1 = lumos(targets)
        y2 = lumos(outputs)
        PSNR.append(-10*np.log10(np.mean((y1-y2)**2)))
        
        if compute_ssim:
            image_ssim = []
            for i in range(targets.shape[0]):
                for rgb in range(3):
                    image_ssim.append(ssim(targets[i,:,:,rgb],outputs[i,:,:,rgb]))
            SSIM.append(np.nanmean(image_ssim))
    
    if compute_ssim:
        SSIM = np.nanmean(SSIM)
    else:
        SSIM = None
    
    return np.mean(MSE), np.mean(PSNR), SSIM

def pixelate(data,scale=2,axis=None):
    
    #preprocess the axis info, is none was provided assume we're operating on each axis
    #'axis' should now be a list of each dim number to be processed
    if type(axis) is int:
        axis = [axis]
    if axis is None:
        axis = np.arange(0,data.ndim)
    axis = list(axis)
        
    #process the scale
    #'scale' should now be a list of scales for each dim in 'data'
    if type(scale) is int:
        scale = [scale]*len(axis)
    scale_list = [1]*data.ndim
    for a,s in zip(axis,scale):
        scale_list[a] = s
    scale = scale_list
    
    #determine the shapes for reshaping:
    target_shape,output_shape = [],[]
    for sc, inshp in zip(scale,data.shape):
        target_shape.append(sc)
        target_shape.append(inshp//sc)
        output_shape.append(inshp//sc)
        
    #make sure the axes are divisible by the scale values:
    for i in range(data.ndim):
        assert data.shape[i]%scale[i] == 0, \
        'Axis '+str(i)+' of length '+str(data.shape[i])+' not divisible by '+str(scale[i])
            
    
    #do the averaging:
    data = np.reshape(data,target_shape,order='F')
    data = np.mean(data,axis = tuple(np.arange(0,data.ndim,2))).squeeze()
    data = np.reshape(data,output_shape,order='F')
    
    return data