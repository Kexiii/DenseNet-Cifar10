# -*- coding:utf-8 -*-
__all__=['getDataGenerator']

import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.datasets import cifar10
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def getDataGenerator(train_phase,rescale=1./255):
    """return the data generator that consistently
    generates image batches after data augmentation
    Args:
        train_phase:
            flag variable that denotes whether the data augmentation is 
        applied on the train set or validation set
        rescale:
            rescaling parameter for Keras ImageDataGenerator
    Return:
        keras data generator
    """
    if train_phase == True:
        datagen = ImageDataGenerator(
        rotation_range=0.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=rescale)
    else: 
        #validation
        #only rescaling is applied on validation set
        datagen = ImageDataGenerator(
        rescale=rescale
        )
    
    return datagen


def testDataGenerator(pics_num):
    """visualize the pics after data augmentation
    Args:
        pics_num:
            the number of pics you want to observe
    return:
        None
    """
    
    print("Now, we are testing data generator......")
    
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    
    # Load label names to use in prediction results
    label_list_path = 'datasets/cifar-10-batches-py/batches.meta'
    keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
    datadir_base = os.path.expanduser(keras_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    label_list_path = os.path.join(datadir_base, label_list_path)
    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)
    
    datagen = getDataGenerator(train_phase=True)
    """
    x_batch is a [-1,row,col,channel] np array
    y_batch is a [-1,labels] np array
    """
    figure = plt.figure()
    plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9,hspace=0.5, wspace=0.3)
    for x_batch,y_batch in datagen.flow(x_train,y_train,batch_size = pics_num):
        for i in range(pics_num):
            pics_raw = x_batch[i]
            pics = array_to_img(pics_raw)
            ax = plt.subplot(pics_num//5, 5, i+1)
            ax.axis('off')
            ax.set_title(labels['label_names'][np.argmax(y_batch[i])])
            plt.imshow(pics)
        plt.savefig("./processed_data.jpg")
        break   
    print("Everything seems OK...")
            
if __name__ == '__main__':
    testDataGenerator(20)