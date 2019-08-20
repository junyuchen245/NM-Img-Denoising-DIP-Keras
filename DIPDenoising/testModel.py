import os
import sys
sys.path.append('/data/jchen/anaconda3/lib/python3.7/site-packages')
import numpy as np
import keras
import math
#from itertools import zip
from keras.models import Model, load_model
from keras import backend as K

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import shutil
import time
from image_reading import load_image_from_folder, load_test_from_folder
import gc
from keras.utils import to_categorical
from scipy.misc import imsave, imread
import matplotlib.pyplot as plt
from vis.visualization import get_num_filters
from skimage.transform import rescale, resize
from keras.layers import *


def get_seg(input_array):
    seg = np.zeros([1,192,192,1])
    for i in range(192):
        for j in range(192):
            seg[0,i,j,0] = np.argmax([input_array[0,i,j,0], input_array[0,i,j,1], input_array[0,i,j,2]])
    return seg

def RepLayer(stack_size):

    def inner(tensor):
        tensor_org = tensor
        for i in range(stack_size-1):
            tensor = concatenate([tensor, tensor_org], axis=3)
        return tensor
    return inner

def Unet(pretrained_weights = None, input_size = (192,192,1)):
    """ second encoder for ct image """
    input_img = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(name='conv_ct_32')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1) #192x192

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(name='conv_ct_64')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2) #96x96

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(name='conv_ct_128')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3) #48x48

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(name='conv_ct_256')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4) #24x24

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(name='conv_ct_512')(conv5)
    conv5 = Dropout(0.5)(conv5)
    #pool5_ct = MaxPool2D(pool_size=(2, 2))(conv5) #12x12

    up6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5)) #24x24
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4, up6], axis=3)  # cm: cross modality
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(name='decoder_conv_256')(conv6)

    up7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, up7], axis=3)  # cm: cross modality
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(name='decoder_conv_128')(conv7)

    up8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, up8], axis=3)  # cm: cross modality
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(name='decoder_conv_64')(conv8)

    up9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, up9], axis=3)  # cm: cross modality
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(name='decoder_conv_32')(conv9) 

    conv10 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv2D(filters=1, kernel_size=1,  activation='relu', padding='same', name='conv12')(conv10)

    model = Model(inputs=input_img, outputs=conv11)

    return model

#print('backend')
#print(K.backend())
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    print('GPU Setup done')

#dtype='float16'
#K.set_floatx(dtype)

# default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
#K.set_epsilon(1e-4)

testImg_path = '/netscratch/jchen/NaF_PETCT_processed/'   #'/netscratch/jchen/patient_test/'
model_path = '/netscratch/jchen/boneSPECTSegmentation/experiments_subnet/outputs/'
image_name = 'patient_2_scan_3'
# image_test = load_test_from_folder(testImg_path, (192, 192), HE=False, Truc=False, Aug=False)

image_test = np.load(testImg_path + image_name + '.npz')
image_test = image_test['a']
image_test  = image_test.reshape(image_test.shape[0], 192, 192*2, 1)

print(image_test.shape)

# Training arguments
net = Unet()
print(net.summary())
dip_model = Model(inputs=net.input, outputs=net.outputs)
dip_model.compile(optimizer=Adam(lr=1e-3), loss='mean_absolute_error')

# -------- Testing phase
print(' testing start')
n_batches = 0
bone_label = np.zeros((144,image_test.shape[0],144))
for img_i in range(1):#range(image_test.shape[0]):
    print(img_i)
    img = image_test[300,:,:,:]
    img = img.reshape(1,192,192*2,1)
    imgSPECT = img[:, :, 192:192 * 2, :]

    #normalize image
    imgSPECT = imgSPECT/np.max(imgSPECT)
    idx = np.random.random((192,192))
    idx = idx.reshape(1,192,192,1)
    tmp, row, col, ch = imgSPECT.shape
    mean = 0
    var = 0.001
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (tmp, row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    # add gaussian noise
    noisy = imgSPECT + gauss

    #noisy = resize(noisy, (1, 192 / 4, 192 / 4, 1),order=0,anti_aliasing=False)
    #noisy = resize(noisy, (1, 192 , 192, 1),order=0,anti_aliasing=False)
    for iter_i in range(300):
        img_ones = np.ones_like(idx)
        dip_model.train_on_batch([idx], noisy)
        loss = dip_model.test_on_batch([idx], noisy)
        outputs = dip_model.predict([idx])
        #idx = outputs
        print('loss = '+str(loss))
        if iter_i % 20 == 0:
            print(iter_i)
            img = dip_model.predict([idx])
            plt.subplot(1,3,1)
            plt.axis('off')
            plt.imshow(img[0, :, :, 0], cmap='gray')
            plt.title('denoised image')
            plt.subplot(1,3,2)
            plt.axis('off')
            plt.imshow(noisy[0, :, :, 0], cmap='gray')
            plt.title('noisy input')
            plt.subplot(1,3,3)
            plt.axis('off')
            plt.imshow(imgSPECT[0, :, :, 0], cmap='gray')
            plt.title('noise-free image')
            plt.savefig('out.png')
            plt.close()
            a = input('enter')


