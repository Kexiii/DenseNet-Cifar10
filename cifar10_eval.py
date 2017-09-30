# -*- coding:utf-8 -*-
import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.datasets import cifar10
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from data_input.data_input import getDataGenerator
from model.DenseNet import createDenseNet
from cifar10_train import check_point_file
from cifar10_train import nb_classes
from cifar10_train import img_dim
from cifar10_train import densenet_depth
from cifar10_train import densenet_growth_rate

batch_size = 128

def eval_model():
    model = createDenseNet(nb_classes=nb_classes,img_dim=img_dim,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)
    model.load_weights(check_point_file)
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    label_list_path = 'datasets/cifar-10-batches-py/batches.meta'   
    keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
    datadir_base = os.path.expanduser(keras_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    label_list_path = os.path.join(datadir_base, label_list_path)
    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)
    
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test= keras.utils.to_categorical(y_test, nb_classes)
    test_datagen = getDataGenerator(train_phase=False)
    test_datagen = test_datagen.flow(x_test,y_test,batch_size = batch_size,shuffle=False)
    
    # Evaluate model with test data set and share sample prediction results
    evaluation = model.evaluate_generator(test_datagen,
                                        steps=x_test.shape[0] // batch_size,
                                        workers=4)
    print('Model Accuracy = %.2f' % (evaluation[1]))
    
    counter = 0
    figure = plt.figure()
    plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9,hspace=0.5, wspace=0.3)
    for x_batch,y_batch in test_datagen:
        predict_res = model.predict_on_batch(x_batch)
        for i in range(batch_size):
            actual_label = labels['label_names'][np.argmax(y_batch[i])]
            predicted_label = labels['label_names'][np.argmax(predict_res[i])]
            if actual_label != predicted_label:
                counter += 1
                pics_raw = x_batch[i]
                pics_raw *= 255
                pics = array_to_img(pics_raw)
                ax = plt.subplot(25//5, 5, counter)
                ax.axis('off')
                ax.set_title(predicted_label)
                plt.imshow(pics)
            if counter >= 25:
                plt.savefig("./wrong_predicted.jpg")
                break
        if counter >= 25:
                break
    print("Everything seems OK...")


if __name__ =='__main__':
    eval_model()