"""
changes 10-2-17
1.  rgbl --> coml
2.  roiflag --> 1 from 0
3.  cls_fr_nbr --> 0 from 480
4.  comment:  #rgbloc, lwfloc, swfloc = get_range_crop_locs(imgrange)
5.  change:
FROM:
    # RGB---------------------------------------------------------------
    if (ver_name[0]=='0'):
        rgbResized = np.zeros((192, 192, 3))
    else:
        imgv_arfobj = ARF.open(Vprobe)
        # arr = imgv_arfobj.load(cls_fr_nbr)
        arr = np.zeros((imgv_arfobj.num_rows ,imgv_arfobj.num_cols, 3))
        # Convert ARF Bayer to RGB
        arr = (arr >> 4)
        #rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_BG2RGB)
        rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_RG2RGB)
TO:
    # RGB---------------------------------------------------------------
    if (ver_name[0]=='0'):
        rgbResized = np.zeros((192, 192, 3))
    else:
        imgv_arfobj = ARF.open(Vprobe)
        # arr = imgv_arfobj.load(cls_fr_nbr)
        #print(imgv_arfobj.num_cols)
        arr = np.zeros((imgv_arfobj.num_rows ,imgv_arfobj.num_cols, 3))
        #print('******************', arr.shape)
        for i in range(3):
            arr[:,:,i] = imgv_arfobj.load(cls_fr_nbr,i)
        #something = mpimg.imread(arr)
        #plt.imshow(something)
        plt.imshow(arr)
        plt.show()
        #input()
        # Convert ARF Bayer to RGB
        # arr = (arr >> 4)
        #rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_BG2RGB)
        # rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_RG2RGB)
6.  change:
FROM:
    # ProbeFolder = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/'
TO:
    ProbeFolder = '/image/Spool/Face/'
7.  change:
FROM:
# wrfilename = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/FaceRecogReport.csv'
TO:
wrfilename = '/image/Spool/Face/FaceRecogReport.csv'

##################  UNDID THESE  ###################################
8.  change:
FROM:
num_classes = labels_test.shape[1]   # = 100
TO:
num_classes = 1 #labels_test.shape[1]   # = 100
9.
FROM:
    #labelsL.append(np.eye(nbr_classes)[clsnbr])
TO:
    labelsL.append(np.eye(nbr_classes)[0])
10. change:
FROM:
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
TO:
        #last_chk_pathQ = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        #last_chk_path = '/home/exx/Dreamcatcher/DemoTrain7Ver3/' + last_check_pathQ
"""


# coding: utf-8

# Same as Ver5 except:
# - add Conf calc
# - operate on ROI provided images - no xxxxloc stuff
# - output data as csv or mat 

"""
#similar for rgb, lar, & sar
        if roiflag:   # images provided are already the correct ROI
            rgbC = rgb
        else:         # need to get correct ROI from entire image
            rgbC = rgb[rgbloc[0]:rgbloc[1], rgbloc[2]:rgbloc[3]] 
"""

# # To Do:
# -  input param = ver_name
# -  change ARF name to standard name in "ProbeFolder" for Vprobe = , Lprobe = , & Sprobe = 
# -  fix cls_fr_nbr
# -  delete path, class fr nbr comment cell
# -  delete face cropping code
# -  delete: def get_range_crop_locs(imrange):
# -  if a mode is missing, need to fix imgid

# 
# ### File Description - Infer / evaluate test / probe image(s)
#  -  do only a single probe or test image
#  -  same as 1st version, just remove comments, experiments, & plots to make streamlined *.py
#  -  input param: ver_name
#  - --------------------------------------------------
#  - move all def up front
#  - remove most comments & debug
#  -  NOT WORK: --> make main a separate function and call it from a loop going thru all 7
# 

# ### File Description - Infer / evaluate test / probe image(s)
# -  do only a single probe or test image
# -  same as 1st version, just remove comments, experiments, & plots to make streamlined *.py
# -  input param: ver_name
# 

# In[1]:

#ver_name = 'VL0' #VLS'
nbr_classes = 9  # 0 to 8, 0=NOT CLASS = #people+1,   #100  # 0 to 99
num_channels = 5
cls_fr_nbr = 0

stddev = 5e-2
wd = 0.0
batch_size = 128
stddevlocal3_4 = 0.04
wdlocal3_4 = 0.004
stddevsml = 1/192.0   # CHANGE to size before softmax linear @@@@@@@@@@@@@@@@@@@@@@@@@
wdsml = 0.0

roiflag = 1


# In[2]:

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os

import cv2, os
from PIL import Image
import matplotlib.image as mpimg
import random
import pickle

#import scipy as sp
from scipy.misc import imresize, imsave

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import ARF
import pylab
import sys
from pprint import *
import re
from time import sleep


# In[3]:

def pltit3(img1, img2, img3, inm):
    # plot the images
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,18))
    f.tight_layout()
    #ax1.get_title('orig in', fontsize=30)
    ax1.set_title(inm + 'rgbs', fontsize=30)
    ax1.imshow(img1)
    #ax2.set_title('NOT Det: ' + imagepaths[5], fontsize=30)
    ax2.set_title(inm + 'lwvs', fontsize=30)
    ax2.imshow(img2, cmap='gray')
    ax3.set_title(inm + 'swrl', fontsize=30)
    ax3.imshow(img3, cmap='gray')
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
    plt.show()

def pltit6(img1a, img1b, img1c, img2a, img2b, img2c):
    # plot the images
    #f, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(2, 3, figsize=(20,18))
    f, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(2, 3, figsize=(10,10))
    f.tight_layout()
    #ax1.get_title('orig in', fontsize=30)
    ax1.set_title('Probe: ' + 'rgbs', fontsize=30)
    ax1.imshow(img1a)
    ax2.set_title('Probe: ' + 'lwvs', fontsize=30)
    ax2.imshow(img1b, cmap='gray')
    ax3.set_title('Probe: ' + 'swrl', fontsize=30)
    ax3.imshow(img1c, cmap='gray')

    ax4.set_title('Gallery: ' + 'rgbs', fontsize=30)
    ax4.imshow(img2a)
    ax5.set_title('Gallery: ' + 'lwvs', fontsize=30)
    ax5.imshow(img2b, cmap='gray')
    ax6.set_title('Gallery: ' + 'swrl', fontsize=30)
    ax6.imshow(img2c, cmap='gray')

    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
    #plt.subplots_adjust(left=0.0, right=0.5, top=0.5, bottom=0.0)
    #plt.show()
    plt.show(block=False)
    #print('START')
    sleep(3)
    #print('STOP')
    plt.close(f)
    #plt.clf()
    #plt.cla()

# In[4]:

def scale3D(img):
    #print(img[:,:,0])
    # The following scales 0 to 1
    #img = np.float32(img)
    img = img.astype(float)
    imgout = np.zeros_like(img)
    imgout[:,:,0] = (img[:,:,0] - np.min(img[:,:,0])) / (np.max(img[:,:,0]) - np.min(img[:,:,0]))
    imgout[:,:,1] = (img[:,:,1] - np.min(img[:,:,1])) / (np.max(img[:,:,1]) - np.min(img[:,:,1]))
    imgout[:,:,2] = (img[:,:,2] - np.min(img[:,:,2])) / (np.max(img[:,:,2]) - np.min(img[:,:,2]))
    return imgout


# In[5]:

def scale1D(img):
    #img = np.float32(img)
    img = img.astype(float)
    imgout = np.zeros_like(img)
    # The following scales 0 to 1
    imgout = (img - np.min(img)) / (np.max(img) - np.min(img))
    return imgout


# In[6]:

def get_range_crop_locs(imgrange):
    if (int(imgrange)==1):
        #print('Range 1 = 23 m')
        #rgbloc = [150:775, 225:1000]
        #lwfloc = [150:290, 325:450]
        #swfloc = [180:290, 225:375]
        rgbloc = [150, 775, 225, 1000]
        lwfloc = [150, 290, 325, 450]
        swfloc = [180, 290, 225, 375]
    elif (int(imgrange)==2):
        #print('Range 2 = 50 m')
        #rgbloc = [200:800, 250:1000]
        #lwfloc = [200:275, 300:360]
        #swfloc = [225:275, 280:350]
        rgbloc = [200, 800, 250, 1000]
        lwfloc = [200, 275, 300, 360]
        swfloc = [225, 275, 280, 350]
    elif (int(imgrange)==3):
        #print('Range 3 = 75 m')
        #rgbloc = [250:750, 250:950]
        #lwfloc = [230:265, 275:350]
        #swfloc = [225:275, 300:350]
        rgbloc = [250, 750, 250, 950]
        lwfloc = [230, 265, 275, 350]
        swfloc = [225, 275, 300, 350]
    elif (int(imgrange)==4):
        #print('Range 4 = 100 m')
        #rgbloc = [300:700, 450:900]
        #lwfloc = [225:260, 275:330]
        #swfloc = [240:270, 310:340]
        rgbloc = [300, 700, 450, 900]
        lwfloc = [225, 260, 275, 330]
        swfloc = [240, 270, 310, 340]
        
    return rgbloc, lwfloc, swfloc


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ###  @@@@@@@@@------------------------------------------------------------@@@@@@@@@
# ###  @@@@@@@@@------------------------------------------------------------@@@@@@@@@
# ###  @@@@@@@@@------- MAIN PROCESSING  ----------------------@@@@@@@@@
# ###  @@@@@@@@@------------------------------------------------------------@@@@@@@@@
# ###  @@@@@@@@@------------------------------------------------------------@@@@@@@@@
# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ###  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# In[7]:

def mainprobe(ver_name):
    
    # Pre-processing
    # train: crop; random flip horizontally; random adjust: hue, contrast, saturation, ?random invert
    # test: crop

    # does one image at a time
    def pre_process_image(image, training):
        # takes single image as input & boolean to do train (True) or test (False)
        if training:
            #print('TRAINING PATH:')
            # Random image crop
            # image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels]))
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random adjust hue, constrast, and saturation
            ###image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            ###image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of above may overflow.  So limit the range
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # Testing
            # crop image around center, then has same size as training images
            # image = tf.image.resize_image_with_crop_or_pad(image, 
                            # target_height=img_size_cropped, 
                            # target_width = img_size_cropped)
            #print('TESTING PATH: Do nothing if not cropping')
            print('Run Inference')
        return image
    
    # The above function is called for each image for the input batch using the following function
    # The way that the training data set is increased / augmented is that:
    # - start with num_samples (say = 5000) in your entire training set
    # - for each batch a group (say 64) images are randomly selected
    # - if you did NOT do the above random adjustments, you only have num_samples=5000 unique images
    #   to train on.
    # - However, b/c each time you select an image, you randomly change it, giving you a new image
    # - Thus it is conceivably possible that you never use one of the original num_samples=5000
    #   training images, but each time it is a new image.  Also, you may never get the same image
    #   again (not more than once).  B/c each time through the selection-randomization process it is
    #   altered=different!
    def pre_process(images, training):
        images = tf.map_fn(lambda image: pre_process_image(image, training), images)
        return images
    
    # Function to build the network model
    def main_network(images, training):
        #print('$$$$$$$$$$$$$$$$$$$$$', images.shape)
        #print('images shape: ',images.get_shape())
        # ********************* prettytensor method ************************
        '''
        x_pretty = pt.wrap(images)
        # pt uses special numbers to distinguish between train & test
        if training:
            phase = pt.Phase.train
        else:
            phase = pt.Phase.infer

        with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
            y_pred, loss = x_pretty.\
                conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
                max_pool(kernel=2, stride=2).\
                flatten().\
                fully_connected(size=256, name='layer_fc1').\
                fully_connected(size=128, name='layer_fc2').\
                softmax_classifier(num_classes=num_classes, labels=y_true)
        return y_pred, loss
        '''

        # ********************* tensorflow Hvass version method **************************
        # Based on tensorflow CIFAR tutorial: changed to match Hvass version
        # Changed softmax method
        # Changed ksize on pooling from [1,3,3,1] to [1,2,2,1]
        # Changed sizes of fully connected nets
        # Removed name=scope.name
        # Add back in scope
        # commented out weight decay
        #
        #

        # conv192 ------------------------------------------------------------
        with tf.variable_scope('conv192') as scope:
            kernel = tf.get_variable(name='weights', 
                             shape=[5, 5, num_channels, 64], 
                             initializer=tf.truncated_normal_initializer(stddev=stddev, 
                                                                     dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
            biases = tf.get_variable(name='biases', 
                                     shape=[64], 
                                     initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv192 = tf.nn.relu(pre_activation) #, name=scope.name)
            #_activation_summary(conv1)

        # pool192 ------------------------------------------------------------
        pool192 = tf.nn.max_pool(conv192, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool1')

        # norm192 ------------------------------------------------------------
        norm192 = tf.nn.lrn(pool192, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='norm1')
        # / 9 --> / num_classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



        # conv96 ------------------------------------------------------------
        with tf.variable_scope('conv96') as scope:
            kernel = tf.get_variable(name='weights', 
                             shape=[5, 5, 64, 64], 
                             initializer=tf.truncated_normal_initializer(stddev=stddev, 
                                                                     dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            conv = tf.nn.conv2d(norm192, kernel, [1,1,1,1], padding='SAME')
            biases = tf.get_variable(name='biases', 
                                     shape=[64], 
                                     initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv96 = tf.nn.relu(pre_activation) #, name=scope.name)
            #_activation_summary(conv1)

        # pool96 ------------------------------------------------------------
        pool96 = tf.nn.max_pool(conv96, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool1')

        # norm96 ------------------------------------------------------------
        norm96 = tf.nn.lrn(pool96, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='norm1')
        # / 9 --> / num_classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    



            # conv48 ------------------------------------------------------------
        with tf.variable_scope('conv48') as scope:
            kernel = tf.get_variable(name='weights', 
                            shape=[5, 5, 64, 64], 
                            initializer=tf.truncated_normal_initializer(stddev=stddev, 
                                                                    dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            conv = tf.nn.conv2d(norm96, kernel, [1,1,1,1], padding='SAME')
            biases = tf.get_variable(name='biases', 
                                     shape=[64], 
                                     initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv48 = tf.nn.relu(pre_activation) #, name=scope.name)
            #_activation_summary(conv1)

        # pool48 ------------------------------------------------------------
        pool48 = tf.nn.max_pool(conv48, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool1')

        # norm48 ------------------------------------------------------------
        norm48 = tf.nn.lrn(pool48, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='norm1')
        # / 9 --> / num_classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    





        # conv1 ------------------------------------------------------------
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable(name='weights', 
                            shape=[5, 5, 64, 64], 
                            initializer=tf.truncated_normal_initializer(stddev=stddev, 
                                                                    dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            conv = tf.nn.conv2d(norm48, kernel, [1,1,1,1], padding='SAME')
            biases = tf.get_variable(name='biases', 
                                     shape=[64], 
                                     initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation) #, name=scope.name)
            #_activation_summary(conv1)

        # pool1 ------------------------------------------------------------
        pool1 = tf.nn.max_pool(conv1, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool1')

        # norm1 ------------------------------------------------------------
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='norm1')
        # / 9 --> / num_classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # conv2 ------------------------------------------------------------
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable(name='weights', 
                            shape=[5, 5, 64, 64], 
                            initializer=tf.truncated_normal_initializer(stddev=stddev, 
                                                                    dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')
            biases = tf.get_variable(name='biases', 
                                     shape=[64], 
                                     initializer=tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation) #, name=scope.name)
            #_activation_summary(conv2)

        # norm2 ------------------------------------------------------------
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='norm2')
        # / 9 --> / num_classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # pool2 ------------------------------------------------------------
        pool2 = tf.nn.max_pool(norm2, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool2')

        # local3 ------------------------------------------------------------
        with tf.variable_scope('local3') as scope:

            # Move everything into depth so we can perform  a single matrix multiply
            #reshape = tf.reshape(pool2, [batch_size, -1])
            #dim = reshape.get_shape()[1].value
            #dim_reshape1 = dim
            #dim_reshape = reshape.get_shape()
            #dim_pool2 = pool2.get_shape()
            #print('dim_reshape[1]: ********************', dim_reshape1)
            #print('dim_reshape: ********************', dim_reshape)
            #print('dim_pool2: ********************', dim_pool2)

            dimpool2 = pool2.get_shape().as_list()
            #print('dim_pool2: ********************', dimpool2)
            #dim = 6*6*64
            dim = dimpool2[1]*dimpool2[2]*dimpool2[3]
            #print('dim = ', dim)
            #reshape = tf.reshape(pool2, [-1, 6*6*64])
            reshape = tf.reshape(pool2, [-1, dim])
            #reshape = tf.reshape(pool2, [-1, dimpool2[1]*dimpool2[2]*dimpool2[3]])
            #print('reshape shape: ',reshape.get_shape())

            weights = tf.get_variable(name='weights', 
                         shape=[dim, 1024],      #shape=[dim, 384], 
                         initializer=tf.truncated_normal_initializer(stddev=stddevlocal3_4, 
                                                                         dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wdlocal3_4, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable(name='biases', 
                                     shape=[1024],             # shape=[384], 
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases) #, name=scope.name)
            #_activation_summary(local3)

        # local4 ------------------------------------------------------------
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable(name='weights', 
                         shape=[1024, 512],      #shape=[384, 192], 
                         initializer=tf.truncated_normal_initializer(stddev=stddevlocal3_4, 
                                                                         dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wdlocal3_4, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            biases = tf.get_variable(name='biases', 
                                     shape=[512],             #shape=[192], 
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases) #, name=scope.name)
            #_activation_summary(local4)

        # Linear layer ------------------------------------------------------------
        # linear layer (WX + b)
        # Don't apply softmax b/c 
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts unscaled logits
        # and performs the softmax internally for efficiency
        with tf.variable_scope('linear_pre_softmax') as scope:
            weights = tf.get_variable(name='weights', 
                         shape=[512, num_classes],       #shape=[192, num_classes], 
                         initializer=tf.truncated_normal_initializer(stddev=stddevsml, 
                                                                    dtype=tf.float32))
            #if wd is not None:
            #    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wdsml, name='weight_loss')
            #    tf.add_to_collection('losses', weight_decay)    
            biases = tf.get_variable(name='biases', 
                                     shape=[num_classes], 
                                     initializer=tf.constant_initializer(0.0))
            linear_pre_softmax = tf.add(tf.matmul(local4, weights), biases) #, name=scope.name)

        # Softmax layer ------------------------------------------------------------
        #with tf.variable_scope('softmax_layer') as scope:
        y_pred = tf.nn.softmax(linear_pre_softmax)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=linear_pre_softmax, 
                                                                   labels=y_true)
        loss = tf.reduce_mean(cross_entropy)

        # ---------------------------------------------------------------------------
        #return softmax_linear
        return y_pred, loss
    
    #def create_network(training):
    def create_network(training):
        # wrap the nn in the scope named 'network'
        # create new variable during training, and re-use during testing
        with tf.variable_scope('network', reuse=not training):
            # just rename the input placeholder variable for convenience
            images = x #images_train
            #print(images.get_shape())
            # Create TF graph for pre-processing
            images = pre_process(images=images, training=training)
            # Create TF graph for main processing
            #print(images.get_shape())
            #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            y_pred, loss = main_network(images=images, training=training)
        return y_pred, loss

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    # ARF Probes
    # ProbeFolder = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/'
    ProbeFolder = '/image/Spool/Face/'
    
    img_fns = [os.path.join(ProbeFolder, f) for f in os.listdir(ProbeFolder)]
    #print('@@@@@@@@@@@@@@ HERE @@@@@@@@@@@@@@@@@@@', img_fns)
    for img_fn in img_fns:
        if ( os.path.isfile(img_fn) ):
            #print('@@@@@@@@@@@@@@ HERE @@@@@@@@@@@@@@@@@@@', img_fn)
            if 'coml' in img_fn:    #'rgbl' in img_fn:
                Vprobe = img_fn
            elif 'lwvl' in img_fn:
                Lprobe = img_fn
            elif 'swrl' in img_fn:
                Sprobe = img_fn
            else:
                print('********** Bad File Names **********')
                break

    #print(Vprobe)
    #print(Lprobe)
    #print(Sprobe) 

    # ASSUME:  RGB, LW, & SW are all at same range!
    imgid = Vprobe[-14:]
    imgrange = imgid[2]
    imgclass = imgid[3:5]
    #print('img file nam e: ', Vprobe)
    #print('imgid: ', imgid)
    #print('range: ', imgrange)
    #print('class: ', imgclass)
    #rgbloc, lwfloc, swfloc = get_range_crop_locs(imgrange)

    # RGB---------------------------------------------------------------
    if (ver_name[0]=='0'):
        rgbResized = np.zeros((192, 192, 3))
    else:
        imgv_arfobj = ARF.open(Vprobe)
        # arr = imgv_arfobj.load(cls_fr_nbr)
        #print(imgv_arfobj.num_cols)
        arr = np.zeros((imgv_arfobj.num_rows ,imgv_arfobj.num_cols, 3))
        #print('******************', arr.shape)
        #for i in range([0, 1, 2]):
        #    arr[:,:,i] = imgv_arfobj.load(cls_fr_nbr,i)

        arr[:,:,0] = imgv_arfobj.load(cls_fr_nbr,0)
        arr[:,:,1] = imgv_arfobj.load(cls_fr_nbr,1)
        arr[:,:,2] = imgv_arfobj.load(cls_fr_nbr,2)
        rgb = scale3D(arr)

        #something = mpimg.imread(arr)
        #plt.imshow(something)
        #plt.imshow(arr)
        #plt.show()
        #input()
        # Convert ARF Bayer to RGB
        # arr = (arr >> 4)
        #rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_BG2RGB)
        # rgb = cv2.cvtColor(arr,cv2.COLOR_BAYER_RG2RGB)

        # # reshape to standard 192x192 arrays and Combine the 3 images
        if roiflag:   # images provided are already the correct ROI
            rgbC = rgb
        else:         # need to get correct ROI from entire image
            rgbC = rgb[rgbloc[0]:rgbloc[1], rgbloc[2]:rgbloc[3]] 
        rgbResized = imresize(rgbC, [192,192,3])             
        #rgbResized = rgbResized/np.max(rgbResized)
        rgbResized = scale3D(rgbResized)
        #print('rgb real:', rgbResized.shape)    

    # LWVL---------------------------------------------------------------
    if (ver_name[1]=='0'):
        larResized = np.zeros((192, 192, 1))
    else:
        imgl_arfobj = ARF.open(Lprobe)
        lar = imgl_arfobj.load(cls_fr_nbr)
        if roiflag:   # images provided are already the correct ROI
            larC = lar
        else:         # need to get correct ROI from entire image
            larC = lar[lwfloc[0]:lwfloc[1], lwfloc[2]:lwfloc[3]]
        larResized = imresize(larC, [192,192])
        #larResized = larResized/np.max(larResized)
        larResized = scale1D(larResized)
        larResized = np.expand_dims(larResized, 3)
        #print('l real:', larResized.shape)

    # SWRL---------------------------------------------------------------
    if (ver_name[2]=='0'):
        sarResized = np.zeros((192, 192, 1))
    else:
        imgs_arfobj = ARF.open(Sprobe)
        sar = imgs_arfobj.load(cls_fr_nbr)
        if roiflag:   # images provided are already the correct ROI
            sarC = sar
        else:         # need to get correct ROI from entire image
            sarC = sar[swfloc[0]:swfloc[1], swfloc[2]:swfloc[3]]
        sarResized = imresize(sarC, [192,192])
        #sarResized = sarResized/np.max(sarResized)
        sarResized = scale1D(sarResized)
        sarResized = np.expand_dims(sarResized, 3)
        #print('s real:', sarResized.shape)

    imgvls = np.concatenate((rgbResized, larResized, sarResized), axis=2)
    save_pathnpy = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/'
    save_name = save_pathnpy + ver_name + 'W' + imgid[:-4] + str(cls_fr_nbr) +'.npy'
    #pltit3(imgvls[:,:,:3], imgvls[:,:,3], imgvls[:,:,4], 'Probe: ')
    np.save(save_name, imgvls)

    """
    # Verify
    imgvlsback = np.load(save_name)
    print('rgbResized')
    print(rgbResized.shape, rgbResized.dtype, type(rgbResized), np.min(rgbResized), np.max(rgbResized))
    print('larResized')
    print(larResized.shape, larResized.dtype, type(larResized), np.min(larResized), np.max(larResized))
    print('sarResized')
    print(sarResized.shape, sarResized.dtype, type(sarResized), np.min(sarResized), np.max(sarResized))
    print(imgvls.shape, imgvls.dtype, type(imgvls), np.min(imgvls), np.max(imgvls))
    print(save_name)
    print(imgvlsback.shape, imgvlsback.dtype, type(imgvlsback), np.min(imgvlsback), np.max(imgvlsback))
    print(imgvls.shape[0]*imgvls.shape[1]*imgvls.shape[2])
    print(sum(sum(sum(imgvls == imgvlsback))))
    #
    print('imgvlsback[:,:,:3] rgbResized')
    print(imgvlsback[:,:,:3].shape, imgvlsback[:,:,:3].dtype, type(imgvlsback[:,:,:3]), np.min(imgvlsback[:,:,:3]), np.max(imgvlsback[:,:,:3]))
    print('imgvlsback[:,:,0] rgbResized')
    print(imgvlsback[:,:,0].shape, imgvlsback[:,:,0].dtype, type(imgvlsback[:,:,0]), np.min(imgvlsback[:,:,0]), np.max(imgvlsback[:,:,0]))
    print('imgvlsback[:,:,1] rgbResized')
    print(imgvlsback[:,:,1].shape, imgvlsback[:,:,1].dtype, type(imgvlsback[:,:,1]), np.min(imgvlsback[:,:,1]), np.max(imgvlsback[:,:,1]))
    print('imgvlsback[:,:,2] rgbResized')
    print(imgvlsback[:,:,2].shape, imgvlsback[:,:,2].dtype, type(imgvlsback[:,:,2]), np.min(imgvlsback[:,:,2]), np.max(imgvlsback[:,:,2]))
    # 
    print('imgvlsback[:,:,3] larResized')
    print(imgvlsback[:,:,3].shape, imgvlsback[:,:,3].dtype, type(imgvlsback[:,:,3]), np.min(imgvlsback[:,:,3]), np.max(imgvlsback[:,:,3]))
    print('imgvlsback[:,:,4] sarResized')
    print(imgvlsback[:,:,4].shape, imgvlsback[:,:,4].dtype, type(imgvlsback[:,:,4]), np.min(imgvlsback[:,:,4]), np.max(imgvlsback[:,:,4]))        
    #
    pltit3(imgvlsback[:,:,:3], imgvlsback[:,:,3], imgvlsback[:,:,4])
    input()
    """

    # make a list of filenames, classes, & labels
    imgs_fnL = []
    classesL = []
    labelsL = []
    imgs_fnL.append(save_name)

    # Get class number
    #print(img_fn_v)
    clsnbr = np.uint8(imgid[-11:-9])
    classesL.append(clsnbr)
    #print('=====================', clsnbr,type(clsnbr))
    # Do class label
    labelsL.append(np.eye(nbr_classes)[clsnbr])
    #labelsL.append(np.eye(nbr_classes)[2])

    #print()
    #print('classesL: ', classesL)
    #print('labelsL: ', labelsL)
    #print('imgs_fnL: ', imgs_fnL)

    images_test = np.asarray(imgs_fnL)
    cls_test = np.asarray(classesL)
    labels_test = np.asarray(labelsL)

    class_names = []
    for ii in range(len(labels_test[0])):
        class_names.append(str(ii))
    #print(class_names)

    imgw = imgvls  #np.load(imgs_fnL[0])
    #print(imgw)
    img1 = imgw[:,:,:3]
    img2 = imgw[:,:,3]
    img3 = imgw[:,:,4]

    #print(img1.shape, img1.dtype, type(img1), np.min(img1), np.max(img1))
    #print(img2.shape, img2.dtype, type(img2), np.min(img2), np.max(img2))
    #print(img3.shape, img3.dtype, type(img3), np.min(img3), np.max(img3))

    ### The probe image here:
    img_size_row = imgw.shape[0]
    img_size_col = imgw.shape[1]
    num_classes = labels_test.shape[1]   # = 100

    # Set up TF Graph and Session
    tf.reset_default_graph()
    # Placeholders = input variables
    # input images
    x = tf.placeholder(tf.float32, shape=[None, img_size_row, img_size_col, num_channels], name='x')
    # input true labels = one-hot
    #y_true = y_true_labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    # input true class number = integer
    y_true_cls = tf.arg_max(y_true, dimension=1)
    #y_true_cls = cls_train or cls_test

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    # create the nn for training
    _, loss = create_network(training=True)  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

    # Create network for test phase: y_pred is one-hot
    y_pred, _ = create_network(training=False) #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Calc predicted class number as an integer
    y_pred_cls = tf.arg_max(y_pred, dimension=1)

    # Vector of booleans telling whether predicted class equals true class, for each image
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    # Calc classification accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()

    # Saver
    saver = tf.train.Saver()

    #save_dir = 'checkpointsVLS/'       # generic
    #save_dir = 'checkpointsVL0/'       # just 1st 5 frames of each object
    #save_dir = 'checkpointsVL0_1-3rd/'   # sample every 3rd frame
    #save_dir = 'checkpointsVL0_10th/'   # sample every 10th frame
    #save_dir = 'checkpointsVL0_300th_RT/'   # sample every 300th frame but fetch images in Real Time
    #save_dir = 'checkpointsVL0_300th_RT/'   # sample every 300th frame but fetch images in Real Time

    # sample every xth frame but fetch images in Real Time
    #save_dir = 'checkpoints' + ver_name + '_3/'
    save_dir = '/home/exx/Dreamcatcher/DemoTrain7Ver3/' + 'checkpoints' + ver_name + '_3/'
    #print(save_dir)
    if not os.path.exists(save_dir):
        #print('baddddddddddddddddddd')
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'FaceRecog')
    #print(save_path)

    try:
        print('Trying to restore last checkpoint ...')
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        #print('*****************************', last_chk_path)
        #last_chk_pathQ = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        #last_chk_path = '/home/exx/Dreamcatcher/DemoTrain7Ver3/' + last_chk_pathQ
        saver.restore(session, save_path=last_chk_path)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Restored checkpoint from:', last_chk_path)
    except:
        last_chk_pathQ = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        print('save_dir', save_dir)
        print('*****************************QQQQQQQQQQQQQQQQ', last_chk_pathQ)
        #last_chk_path = '/home/exx/Dreamcatcher/DemoTrain7Ver3/' + last_check_pathQ
        #print('************************************************', last_chk_path)
        print('Failed to restore checkpoint.  Initializing variables instead.')
        session.run(tf.global_variables_initializer()) 

    probeimg = np.load(images_test[0])
    probeimg = np.asarray(probeimg)
    probeimg = np.expand_dims(probeimg, 0)
    #print(probeimg.shape)

    # get predicted class label and class number for this image
    #label_pred, cls_pred = session.run([y_pred, y_pred_cls], feed_dict={x: [images_test]})
    label_pred, cls_pred = session.run([y_pred, y_pred_cls], feed_dict={x: probeimg})

    #print(cls_pred)
    #print(label_pred)
    # see index 3 for label_pred below.  it is ~ 0.99999
    # all other indices are vary small
    # tf.??.top_rank ???

    #pred_cls_nbr[i:j] = session.run(tf.argmax(label_pred, 1))
    values, indices = session.run(tf.nn.top_k(label_pred, 3))

    ## To interpret the following results:
    #- look at the "indices:"
    #- each row gives the top 3 ranking for that image, concretely
    #    - 1st row --> top 3 ranks are classes 5, 2, & 4
    #    - 2nd row --> top 3 ranks are classes 5, 7, & 2
    #    - 3rd row --> top 3 ranks are classes 2, 5, & 1
    #    - etc
    #- the "values:", gives the associated probabilities with these class rankings

    #print('pred_cls_nbr', pred_cls_nbr)
    #print('********** top 3 rank: **********')   #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #print('values: \n', values)                  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #print('Classes: \n', indices)                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #print('softmax_cls_pred', softmax_cls_pred)

    cls_str = '0' + str(indices[0][0]) + '_'
    #print(cls_str)

    #Gallery = '/home/exx/Dreamcatcher/DemoTrain7Ver3/pickle_files/' \
    #                + ver_name + '_images_train.pickle'
    Gallery = '/home/exx/Dreamcatcher/DemoTrain7Ver3/pickle_files/' \
                    + 'VLS' + '_images_train.pickle'

    gallery_image = pickle.load(open(Gallery, 'rb'))
    #print(gallery_image)
    for fn in gallery_image:
        #print(fn)
        if cls_str in fn:
            img = np.load(fn)
            #pltit3(img[:,:,:3], img[:,:,3], img[:,:,4], "Gallery: ")
            break

    #pltit3(imgvls[:,:,:3], imgvls[:,:,3], imgvls[:,:,4], 'Probe: ')
    pltit6(imgvls[:,:,:3], imgvls[:,:,3], imgvls[:,:,4], img[:,:,:3], img[:,:,3], img[:,:,4])

    session.close()

    return values, indices


# In[8]:

def write_scores_csv(infile, ver_name, classes, values):
    strTitle ='********** Top 3 rank ' + ver_name + ' **********'
    i=0
    #infile.seek(0)
    infile.write('\n')       
    infile.write(strTitle + '\n')

    for cls in classes:
        strClass = 'Class: ' + str(cls) + ',' + 'Rank' + str(i+1) + ': ' + str(values[i]) + '\n'
        i=i+1;
        infile.write(strClass)


def confcalc(VzzVec):
    loadpath = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/'
    acc = [[0.999, 0.999, 0.995, 0.998, 0.995, 0.991, 0.996]]

    csV00 = np.load(loadpath + 'V00_values.npy')
    cs00S = np.load(loadpath + '00S_values.npy')
    cs0L0 = np.load(loadpath + '0L0_values.npy')
    cs0LS = np.load(loadpath + '0LS_values.npy')
    csVL0 = np.load(loadpath + 'VL0_values.npy')
    csV0S = np.load(loadpath + 'V0S_values.npy')
    csVLS = np.load(loadpath + 'VLS_values.npy')

    #print('csVLS shape: ', csVLS.shape, csVLS[:5])

    yV00 = np.mean(csV00)
    y00S = np.mean(cs00S)
    y0L0 = np.mean(cs0L0)
    y0LS = np.mean(cs0LS)
    yVL0 = np.mean(csVL0)
    yV0S = np.mean(csV0S)
    yVLS = np.mean(csVLS)

    unbiased_estV00 = np.var(csV00) / csV00.shape[0]
    unbiased_est00S = np.var(cs00S) / cs00S.shape[0]
    unbiased_est0L0 = np.var(cs0L0) / cs0L0.shape[0]
    unbiased_est0LS = np.var(cs0LS) / cs0LS.shape[0]
    unbiased_estVL0 = np.var(csVL0) / csVL0.shape[0]
    unbiased_estV0S = np.var(csV0S) / csV0S.shape[0]
    unbiased_estVLS = np.var(csVLS) / csVLS.shape[0]

    Tdist = 1.646

    new_varV00 = Tdist * np.sqrt(unbiased_estV00)
    new_var00S = Tdist * np.sqrt(unbiased_est00S)
    new_var0L0 = Tdist * np.sqrt(unbiased_est0L0)
    new_var0LS = Tdist * np.sqrt(unbiased_est0LS)
    new_varVL0 = Tdist * np.sqrt(unbiased_estVL0)
    new_varV0S = Tdist * np.sqrt(unbiased_estV0S)
    new_varVLS = Tdist * np.sqrt(unbiased_estVLS)

    #print('mean: ', yVLS, 'unbiased est: ', unbiased_estVLS, 'new var: ', new_varVLS)

    adj= 1

    cs_thresh = np.zeros((7,1))
    cs_thresh[0] = yV00 - adj*new_varV00
    cs_thresh[1] = y00S - adj*new_var00S
    cs_thresh[2] = y0L0 - adj*new_var0L0
    cs_thresh[3] = y0LS - adj*new_var0LS
    cs_thresh[4] = yVL0 - adj*new_varVL0
    cs_thresh[5] = yV0S - adj*new_varV0S
    cs_thresh[6] = yVLS - adj*new_varVLS

    #print('cs_thresh shape: ', cs_thresh.shape)
    #print('cs_thresh: ', cs_thresh)

    ID = np.transpose(VzzVec[:,1,:])
    cs_probe = np.transpose(VzzVec[:,0,:])

    #print('ID shape: ', ID.shape, 'cs_probe shape: ', cs_probe.shape)
    #print(ID)
    #print(cs_probe)
    
    acc_weights = np.transpose(acc / np.sum(acc))
    #print('acc wts: ', acc_weights.shape, acc_weights)

    probe1 = np.reshape((1.0/2.0) + (cs_probe[0,:]/2.0), (7,1))
    cs_rk1T = np.multiply( np.multiply(cs_thresh, acc_weights), probe1)
    cs_rk1 = np.transpose(cs_rk1T)
    probe2 = np.reshape((1.0/3.0) + (cs_probe[1,:]/3.0), (7,1))
    cs_rk2T = np.multiply( np.multiply(cs_thresh, acc_weights), probe2)
    cs_rk2 = np.transpose(cs_rk2T)
    probe3 = np.reshape((1.0/4.0) + (cs_probe[2,:]/4.0), (7,1))
    cs_rk3T = np.multiply( np.multiply(cs_thresh, acc_weights), probe3)
    cs_rk3 = np.transpose(cs_rk3T)

    #print('probe1 shape: ', probe1.shape)
    #print('cs_probe shape: ', cs_probe.shape)
    #print('cs_probe[0,:] ', cs_probe[0,:])
    #print('probe1: ', probe1)

    #print('cs_rk1: ', cs_rk1)
    #print('cs_rk2: ', cs_rk2)
    #print('cs_rk3: ', cs_rk3)

    #ID[0,:] = np.array([5., 5., 23., 5., 23., 105., 5.])
    #print('ID shape: ', ID[0,:].shape)
    print('ID[0,:] ', ID[0,:])
    print('cs_rk1 shape: ', cs_rk1.shape)
    print(cs_rk1)
    uniIDs = np.unique(ID[0,:])
    """
    for uniID in uniIDs:
        vidx = np.asarray(np.where(uniID==ID[0,:]))
        #print('uniID: ', uniID, 'uniID locs: ', vidx.shape, vidx)
        out_cs = np.sum(cs_rk1[0][vidx])
        #print('out_cs: ', out_cs)
        print('********** Rank1 Class: {}, Confidence Score: {}'.format(uniID, out_cs))
    """
    vidx = np.asarray(np.where(uniIDs[0]==ID[0,:]))
    out_cs = np.sum(cs_rk1[0][vidx])
    print('********** Rank1 Class: {}, Confidence Score: {}'.format(uniIDs[0], out_cs))
    
    return uniIDs[0], out_cs





############################ MAIN #########################################

# Vzz , values , indices
VzzVec = np.zeros((7,2,3))
Vzznm = ['V00', '00S', '0L0', '0LS', 'VL0', 'V0S', 'VLS']

"""
values = np.load('/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/V00_values.npy')
indices = np.load('/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/V00_indices.npy')
print(values)
print(values.shape)
print(values[:5])
print(indices.shape)
print(indices[:5])
"""

#values=probs, indices=class#
VzzVec[0,0,:], VzzVec[0,1,:] = mainprobe('V00')
VzzVec[1,0,:], VzzVec[1,1,:] = mainprobe('00S')
VzzVec[2,0,:], VzzVec[2,1,:] = mainprobe('0L0')
VzzVec[3,0,:], VzzVec[3,1,:] = mainprobe('0LS')
VzzVec[4,0,:], VzzVec[4,1,:] = mainprobe('VL0')
VzzVec[5,0,:], VzzVec[5,1,:] = mainprobe('V0S')
VzzVec[6,0,:], VzzVec[6,1,:] = mainprobe('VLS')

uniID, out_cs = confcalc(VzzVec)

"""
print()
print()
print()
for ii in range(7):
    print('********** top 3 rank', Vzznm[ii], ': **********')
    print('Classes: ', VzzVec[ii,1,:])
    print('values: ', VzzVec[ii,0,:])
    print()
"""


# wrfilename = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/FaceRecogReport.csv'
wrfilename = '/image/Spool/Face/FaceRecogReport.csv'
infile = open(wrfilename,'a+')
"""
for ii, v in enumerate(Vzznm):
    strTitle ='********** Top 3 rank ' + v + ' **********'
    for ii in range(3):
        infile.write(strTitle)
        #write_scores_csv(self,infile, ver_name, classes, values)
        write_scores_csv(infile, v, VzzVec[ii,1,:], VzzVec[ii,0,:])
"""

for ii, v in enumerate(Vzznm):
        #write_scores_csv(self,infile, ver_name, classes, values)
        write_scores_csv(infile, v, VzzVec[ii,1,:], VzzVec[ii,0,:])

infile.write('\n') 
infile.write('\n') 
strTitle ='********** Rank1 Class: and Confidence Score: '
infile.write('\n')       
infile.write(strTitle + '\n')
strClass = 'Class: ' + str(uniID) + ',' + 'Confidence Score: ' + str(out_cs)
infile.write(strClass)

infile.close()



"""
resultpath = '/home/exx/Dreamcatcher/DemoTrain7Ver3/ARF_Probe/Arraysnpy/VzzVec.npy'
np.save(resultpath, VzzVec)

VzzVecBack = np.load(resultpath)
print()
print('VzzVecBack: \n')
print(VzzVecBack)
"""

"""
vV00, iV00 = mainprobe('V00')
#plt.close('all')
#input()
v00S, i00S = mainprobe('00S')
#input()
v0L0, i0L0 = mainprobe('0L0')
#input()
v0LS, i0LS = mainprobe('0LS')
#input()
vVL0, iVL0 = mainprobe('VL0')
#input()
vV0S, iV0S = mainprobe('V0S')
#input()
vVLS, iVLS = mainprobe('VLS')
"""
# In[ ]:




# In[ ]:



