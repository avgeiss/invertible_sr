#neural_networks.py
from keras.layers import AveragePooling2D, Conv2D, concatenate, UpSampling2D, add
from keras.layers import Input, Lambda, LeakyReLU, subtract, Conv2DTranspose
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf



#################################   CONSTANTS   ###############################

LR = 48     #high resolution image size
US = 2      #number of up-samples
CHAN = 3    #number of channels for the image
NETWORK_NAMES = ['srcnn','dun','edrn','dnsr','dbpn','lapsrn','rdnsr']



###########################   CNN COMPONENTS   ################################
#define some commonly used neural network components, some of these are just 
#shorthands to make the network definitions more concise:
    
#basic convolutional layer
def conv(x,channels,FS=3,activation='relu',use_bias=True):
    return Conv2D(int(channels),(FS,FS),padding='same',activation=activation,use_bias=use_bias)(x)

#keras 2D upsampling layer       
def up(x,N=1,interp='nearest'):
    BS = int(2**N)
    return UpSampling2D((BS,BS),interpolation=interp)(x)

#2D average pooling based downsampler
def down(x,N=1):
    BS = int(2**N)
    return AveragePooling2D((BS,BS))(x)

#deconvolutional upsampling layer
def cup(x,channels=0,FS=4,US=1,activation='relu'):
    if not channels:
        channels = x.get_shape().as_list()[-1]
    return Conv2DTranspose(channels,(FS,FS),strides=(2**US,2**US),padding='same',activation=activation)(x)

#strided convolutional downsampling layer
def cdown(x,FS=4,DS=1,activation='relu'):
    channels = x.get_shape().as_list()[-1]
    return Conv2D(channels,(FS,FS),strides=(2**DS,2**DS),padding='same',activation=activation)(x)

#up/down sampling with "pixel-shuffle"
def psup(x,N=1):
    return Lambda(lambda x:tf.nn.depth_to_space(x,2**N))(x)

def psdown(x,N=1):
    return Lambda(lambda x:tf.nn.space_to_depth(x,2**N))(x)

#shorthand for concatenate function
def cat(x):
    return concatenate(x)

#output layer, converts to 3-channel image with values between (-1,1)
def output(x,FS=3):
    return conv(x,CHAN,FS,activation='tanh')

#switch between (0,255) and (-1,1) images:
def to_uint8(x):
    return Lambda(lambda x: 127.5*(x+1.0))(x)

def to_double(x):
    return Lambda(lambda x: x/127.5-1.0)(x)

#downsampling aware operator, forces 2D averaged downsampled HR output to match
#the LR input
def consv(images):
    xout = images[1]
    xin = images[0]
    xbar = up(down(xout,2),2,interp='nearest')     #average the 4x4 pixel blocks
    c = up(xin,2,interp='nearest')
    si = tf.math.sign(c-xbar)
    return xout + (c-xbar)*(si-xout)/(si-xbar)

#returns average value of alpha over an image
def alpha(images):
    xout = images[1]
    xin = images[0]
    xbar = up(down(xout,2),2,interp='nearest')     #average the 4x4 pixel blocks
    c = up(xin,2,interp='nearest')
    si = tf.math.sign(c-xbar)
    return tf.math.reduce_mean(tf.math.abs((c-xbar)/(si-xbar)))



##############################     CNNs     ###################################

#the 9-5-5 version of the srcnn:
def srcnn(x):
    x = up(x,US)
    x = conv(x,64,9)
    x = conv(x,32,5)
    return output(x,5)

#dense u-net
def dun(x):
    C0 = 38     #number of features (growth rate) at input res
    DS = 4      #number of downsamples in u-net
    BS = 3      #convs per dense block
    
    def dense_block(x,channels):
        for i in range(BS):
            x = cat([x,conv(x,channels)])
        return conv(x,channels,FS=1,activation='linear')
    
    def rec_unet(x,chan,ds):
        x = dense_block(x,chan)
        if ds<DS:
            x = dense_block(conv(cat([x,cup(rec_unet(down(x),chan*1.45,ds+1))]),chan),chan)
        return x
    
    x = rec_unet(x,C0,0)
    for i in range(US):
        x = dense_block(cup(x),int(C0*2**(-i-1)))
        
    return output(x)

#enhanced deep residual
def edrn(x):
    B = 16      #res blocks
    BF = 128    #filter count in res blocks
    NF = 256    #filters in input and output layers
    
    def res_block(x_in):
        x = conv(x_in,BF)
        x = conv(x,BF,activation='linear')
        x = Lambda(lambda x: x*0.1)(x)
        return add([x_in,x])
    
    x_in = conv(x,BF)
    x = x_in
    for i in range(B):
        x = res_block(x)
    x = add([x_in,x])
    x = conv(x,NF)
    x = psup(x)
    x = conv(x,NF)
    x = psup(x)
    return output(x)

#dense block based sr network:
def dnsr(x_in):
    NDB = 8  #number of dense blocks (8 in paper)
    DBL = 6  #number of layers per dense block (8 in paper)
    DBC = 16 #number of channels for dense block layers
    C0 = 256 #256 in paper
    
    def dense_block(x):
        for i in range(DBL):
            x = cat([x,conv(x,DBC)])
        return x
    #input feature layer
    x_feat = conv(x_in,C0)
    #dense block section
    x = dense_block(x_feat)
    for i in range(1,NDB):
        x = dense_block(x)
    #compression unit
    x = conv(x,C0,FS=1,activation='linear')
    #upsampling layers
    x = cup(x,channels=C0,FS=3)
    x = cup(x,channels=C0,FS=3)
    return output(x)

#deep back projection network:
def dbpn(x):
    N0 = 256  #number of features in input conv layer
    NR = 64   #number of features in conv blocks
    T = 4     #number of back-projection blocks
    
    def up_proj(x):
        x_in = conv(x,NR,FS=1,activation='linear')
        xhr1 = cup(x_in,FS=8,US=2)
        xlr1 = cdown(xhr1,FS=8,DS=2)
        xhr2 = cup(subtract([xlr1,x_in]),FS=8,US=2)
        return add([xhr2,xhr1])
    
    def down_proj(x):
        x_in = conv(x,NR,FS=1,activation='linear')
        xlr1 = cdown(x_in,FS=8,DS=2)
        xhr1 = cup(xlr1,FS=8,US=2)
        xlr2 = cdown(subtract([xhr1,x_in]),FS=8,DS=2)
        return add([xlr2,xlr1])
    
    x = conv(x,N0)
    x = conv(x,NR,FS=1,activation='linear')
    hr = up_proj(x)
    lr = down_proj(hr)
    hr = cat([up_proj(lr),hr])
    for i in range(T-2):
        lr = cat([down_proj(hr),lr])
        hr = cat([up_proj(lr),hr])
    
    return output(hr)

#laplacian pyramid super resolution network
def lapsrn(x):
    NF = 64 #number of filters
    NL = 10 #number of Conv layers
    def pyr_block(x):
        for i in range(NL):
            x = conv(x,NF,activation='linear')
            x = LeakyReLU(0.2)(x)
        x = cup(x,channels=NF,activation='linear')
        return x
        
    im = cup(x,activation='linear')
    x = pyr_block(x)
    im = add([im,conv(x,3,FS=1,activation='tanh')])
    im = cup(im,activation='linear')
    x = pyr_block(x)
    im = add([im,conv(x,3,FS=1,activation='tanh')])
    
    return output(im)
    
#residual dense network
def rdnsr(x):
    x_in = x
    D = 10  #number of residual dense blocks
    C = 9   #conv layers per dense block
    G = 32  #channels for dense conv layers
    F = 64  #channels for input conv layers
    
    def rdb(x_in):
        channels = x_in.get_shape().as_list()[-1]
        x = cat([x_in,conv(x_in,G)])
        for i in range(C-1):
            x = cat([x,conv(x,G)])
        x = conv(x,channels,FS=1,activation='linear')
        return add([x_in,x])
    #build the network:
    gf = [conv(x_in,F)]
    x = conv(gf[0],F)
    for i in range(D):
        x = rdb(x)
        gf.append(x)
    x = conv(cat(gf),F,FS=1,activation='linear')
    x = add([conv(x,F),gf[0]])
    x = psup(x,N=2)
    return output(x)



########################       BUILD CNNs       ###############################

def build_cnn(name,output_type='tanh',ds_type='avg2d',lr=0.0001):
    #cnn = build_cnn(name,output_type='tanh',ds_type='avg2d',lr=0.0001)
    #
    #function that compiles networks with an appropriate output/loss function.
    #
    #Inputs:
    #   name: String, name of desired CNN. (listed in NETWORK_NAMES variable)
    #   output_type: String, ones of: 
    #           'tanh': conventional tanh transfer function
    #           'strict': Downsampling aware (assumes 2d avg)
    #           'alpha': Downsampling aware with regularized alpha
    #           'l2reg': Conventional tanh output with LR MSE and HR MSE
    #
    #   ds_type: String, downsampling type:
    #           'avg2d': 2d-avg downsampling. note: assumes HR input and output
    #                   and the first layer of the CNN performs the downsampling
    #                   this makes the training-code/data handling simpler. This
    #                   layer can be removed later if needed
    #           'bicubic': Takes a LR input and produces HR output, need to do
    #                   downsampling of training data before hand
    #   lr: Float, learning rate
    #
    #Outputs:
    #   cnn: Compiled keras model
    
    #build the neural network input layers:
    if ds_type=='avg2d':
        HR = LR*2**US                               #infer the output size
        x_in = Input((HR,HR,CHAN))                  #input layer
        x_lr = down(to_double(x_in),N=2)            #tform to -1.0,1.0 pixel space, downsample
    elif ds_type=='bicubic':
        x_in = Input((LR,LR,CHAN))
        x_lr = to_double(x_in)
        
    #pass through the requested CNN architecture:
    x = globals()[name](x_lr)
    
    #apply the output layers:
    if output_type == 'tanh':
        x = to_uint8(x)
        cnn = Model(x_in,x)
        cnn.compile(optimizer=Adam(learning_rate = lr),loss='MSE')
    elif output_type=='strict':
        x = Lambda(consv)([x_lr,x])
        x = to_uint8(x)
        cnn = Model(x_in,x)
        cnn.compile(optimizer=Adam(learning_rate = lr),loss='MSE')
    elif output_type=='alpha':
        a = Lambda(alpha)([x_lr,x])
        x = Lambda(consv)([x_lr,x])
        x = to_uint8(x)
        cnn = Model(x_in,[x,a])
        cnn.compile(optimizer=Adam(learning_rate = lr),loss=['MSE','MSE'],loss_weights=[1,10])
    elif output_type=='l2reg':
        cnn = Model(x_in,[to_uint8(x),to_uint8(down(x,N=2))])
        cnn.compile(optimizer=Adam(learning_rate = lr),loss=['MSE','MSE'],loss_weights=[1,16])
        
    return cnn



##############################   TESTING   ####################################

if __name__ == '__main__':
    for net_name in NETWORK_NAMES:
        cnn = build_cnn(net_name,output_type='strict')
        cnn.summary()
        plot_model(cnn,'./figures/model_diagrams/' + net_name + '.png',show_layer_names=False,show_shapes=True)