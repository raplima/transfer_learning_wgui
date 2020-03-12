# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:07:16 2020

@author: rafael.lima
Helper functions for fine-tunning models
"""

import os
import shutil
import numpy as np
from tensorflow.keras import applications
import tensorflow.keras.backend as K

#from tensorflow.keras.backend import set_session
#from tensorflow.keras.backend import clear_session
#from tensorflow.keras.backend import get_session
#import tensorflow
#import gc

def get_models_dict():
    """
    returns the available CNN models

    Parameters
    ----------

    Returns
    -------
    models_dict : python dictionary
        dictionary containing name and input dimensions for available models.
    """
    
    models_dict = {
        'Xception': (299, 299, 3),
        'VGG16': (224, 224, 3),
        'VGG19': (224, 224, 3),
        'ResNet50': (224, 224, 3),
        'InceptionV3': (299, 299, 3),
        'InceptionResNetV2': (299, 299, 3),
        'MobileNet': (224, 224, 3),
        'MobileNetV2': (224, 224, 3),
        'DenseNet121': (224, 224, 3),
        'NASNetLarge': (331, 331, 3)
    }
    
    return models_dict

def train_valid_test_split(data_dir, val_frac, test_frac = 0.0):
    # for reproducibility
    np.random.seed(1234)
    #print(np.random.randint(1, 10000))  # 8916

    print(f"Splitting data in {data_dir} into training/validation/test")

    # uses the file path and walks in folders to select training and test data
    lst = os.listdir(data_dir)
    # assume elements with "." are files and remove from list
    folders = [item for item in lst if os.path.isdir(os.path.join(data_dir, item))]

    # create folder to save training/validation/test data:
    path_train = os.path.join(os.path.dirname(data_dir), 
                              f'{os.path.basename(data_dir)}_train')
    if os.path.exists(path_train):
        shutil.rmtree(path_train, ignore_errors=True)
    os.makedirs(path_train)

    path_valid = os.path.join(os.path.dirname(data_dir), 
                              f'{os.path.basename(data_dir)}_validation')
    if os.path.exists(path_valid):
        shutil.rmtree(path_valid, ignore_errors=True)
    os.makedirs(path_valid)

    if test_frac > 0.0:
        path_test = os.path.join(os.path.dirname(data_dir), 
                                  f'{os.path.basename(data_dir)}_test')
        if os.path.exists(path_test):
            shutil.rmtree(path_test, ignore_errors=True)
        os.makedirs(path_test)
    else:
        path_test = None

    # for each one of the folders
    for this_folder in folders:
        print(f'----{this_folder}')

        # create folder to save cropped image:
        if os.path.exists(os.path.join(path_train,this_folder)):
            shutil.rmtree(os.path.join(path_train,this_folder), ignore_errors=True)
        os.makedirs(os.path.join(path_train, this_folder))

        if os.path.exists(os.path.join(path_valid,this_folder)):
            shutil.rmtree(os.path.join(path_valid,this_folder), ignore_errors=True)
        os.makedirs(os.path.join(path_valid,this_folder))

        if test_frac > 0.0:
            if os.path.exists(os.path.join(path_test,this_folder)):
                shutil.rmtree(os.path.join(path_test,this_folder), ignore_errors=True)
            os.makedirs(os.path.join(path_test,this_folder))

        # separate training and test data:
        # get pictures in this folder:
        lst = os.listdir(os.path.join(data_dir, this_folder))

        if len(lst) < 3:
            print("      Not enough data for automatic separation")
        else:
            # shuffle the indices:
            np.random.shuffle(lst)

            # number of test images:
            n_valid = np.int(np.round(len(lst) * val_frac))
            n_test = np.int(np.round(len(lst) * test_frac))
            if n_valid == 0:
                print("Small amount of data, only one sample forcefully selected to be part of validation data")
                n_valid = 1
            if test_frac > 0 and n_test == 0:
                print("Small amount of data, only one sample forcefully selected to be part of test data")
                n_test = 1

            # copy all pictures to appropriate folder
            for t in range(0, n_test):
                shutil.copy2(os.path.join(data_dir, this_folder, lst[t]), 
                                          os.path.join(path_test, this_folder))

            for t in range(n_test, n_test + n_valid):
                shutil.copy2(os.path.join(data_dir, this_folder, lst[t]), 
                                          os.path.join(path_valid, this_folder))

            for t in range(n_test + n_valid, len(lst)):
                shutil.copy2(os.path.join(data_dir, this_folder, lst[t]), 
                                          os.path.join(path_train, this_folder))

    print('Split data complete\n')
    
    return path_train, path_valid, path_test


def get_model_memory_usage(batch_size, model):
    #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    #import numpy as np
    #from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output.shape:
            # output_shape of input tensor is a list for some reason:
            if isinstance(s, list):
                s = next(iter(s))
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def model_preprocess(model_name):
    """Loads the appropriate CNN preprocess
      Args:
        arch: String key for model to be loaded.
      Returns:
        The specified Keras preprocessing.
      """
    # function that loads the appropriate model
    if model_name == 'Xception':
        return applications.xception.preprocess_input
    elif model_name == 'VGG16':
        return applications.vgg16.preprocess_input
    elif model_name == 'VGG19':
        return applications.vgg19.preprocess_input
    elif model_name == 'ResNet50':
        return applications.resnet50.preprocess_input
    elif model_name == 'InceptionV3':
        return applications.inception_v3.preprocess_input
    elif model_name == 'InceptionResNetV2':
        return applications.inception_resnet_v2.preprocess_input
    elif model_name == 'MobileNet':
        return applications.mobilenet.preprocess_input
    elif model_name == 'DenseNet121':
        return applications.densenet.preprocess_input
    elif model_name == 'NASNetLarge':
        return applications.nasnet.preprocess_input
    elif model_name == 'MobileNetV2':
        return applications.mobilenet_v2.preprocess_input
    else:
        print('Invalid model selected')
        return False
    
def model_app(arch, input_tensor, weights):
    """Loads the appropriate convolutional neural network (CNN) model
      Args:
        arch: String key for model to be loaded.
        input_tensor: tensor to use as image input for the model.
        weights: one of 'imagenet' or None
      Returns:
        model: The specified Keras Model instance with ImageNet weights loaded and without the top classification layer.
      """
    # function that loads the appropriate model
    if arch == 'Xception':
        model = applications.Xception(weights=weights, include_top=False, input_tensor=input_tensor)
        print('Xception loaded')
    elif arch == 'VGG16':
        model = applications.VGG16(weights=weights, include_top=False, input_tensor=input_tensor)
        print('VGG16 loaded')
    elif arch == 'VGG19':
        model = applications.VGG19(weights=weights, include_top=False, input_tensor=input_tensor)
        print('VGG19 loaded')
    elif arch == 'ResNet50':
        model = applications.ResNet50(weights=weights, include_top=False, input_tensor=input_tensor)
        print('ResNet50 loaded')
    elif arch == 'InceptionV3':
        model = applications.InceptionV3(weights=weights, include_top=False, input_tensor=input_tensor)
        print('InceptionV3 loaded')
    elif arch == 'InceptionResNetV2':
        model = applications.InceptionResNetV2(weights=weights, include_top=False, input_tensor=input_tensor)
        print('InceptionResNetV2 loaded')
    elif arch == 'MobileNet':
        model = applications.MobileNet(input_shape=(224, 224, 3), weights=weights, include_top=False,
                                       input_tensor=input_tensor)
        print('MobileNet loaded')
    elif arch == 'DenseNet121':
        model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('DenseNet121 loaded')
    elif arch == 'NASNetLarge':
        model = applications.NASNetLarge(weights=weights, include_top=False, input_tensor=input_tensor)
        print('NASNetLarge loaded')
    elif arch == 'MobileNetV2':
        model = applications.MobileNetV2(input_shape=(224, 224, 3), weights=weights, include_top=False,
                                         input_tensor=input_tensor)
        print('MobileNetV2 loaded')
    else:
        print('Invalid model selected')
        model = False

    return model