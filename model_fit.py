# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:05:23 2020

@author: rafael.lima

functions to test transfer learning and fine-tunning
This module contains functions:
    make_model:
        assembles the cnn model
    train_model:
        fits the assembled model
"""
import os
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import optimizers 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import pandas as pd
import numpy as np

from helper_functions import get_model_memory_usage, model_app

def make_model(model_name, input_tensor, model_pars, weights):
    """
    Assembles the model with base and top that can be used for fine-tuning. 
    The assembled model contains the base layers frozen

    Parameters
    ----------
    model_name : STRING
        Name of one of the models to be loaded
    input_tensor : TUPLE
        dimensions of model input
    model_pars : PYTHON LIST OF DICTIONARIES
        The layers parameters of the top model. Accepts fully connected or dropout layers. 
        example:
            model_pars = [{'Dropout': {'rate': 0.5}}, 
                          {'Dense':   {'units': 100, 'activation': 'relu'}}, 
                          {'Dense':   {'units': 100, 'activation': 'relu'}}]
    weights : STRING OR NONE
        one of 'imagenet' or None
    Returns
    -------
    model

    """
    
    # load the base model:
    base_model = model_app(model_name, tf.keras.Input(shape=input_tensor), weights)
    # freeze layers (layers will not be updated during the first training process)
    for layer in base_model.layers:
    	layer.trainable = False
    
    # create the top model:
    top_model = Sequential()
    
    top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:4]))
    
    for layer_info in model_pars:
        k = next(iter(layer_info))
        if k == 'Dropout':
            top_model.add(Dropout(**layer_info[k]))
        if k == 'Dense':
            top_model.add(Dense(**layer_info[k]))
    
    # set the entire model:
    # build the network
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    return model
    
def train_model(model_path, 
                train_dir, valid_dir, test_dir, 
                datagen, 
                epochs, 
                batch_size,
                loss, 
                opt, 
                metrics, 
                unfreeze = False, 
                patience = 0):
    """
    function to train a model previously saved as hdf5 file using specified
    parameters

    Parameters
    ----------
    model_path : STRING (os.path)
        model file path
    train_dir : STRING (os.path)
        file path to training directory.
    valid_dir : STRING (os.path)
        file path to validation directory.
    test_dir : STRING (os.path)
        file path to test directory.
    datagen : ImageDataGenerator 
        keras ImageDataGenerator instance
    epochs : INT
        number of epochs the model will be trained for.
    batch_size : INT
        batch_size used for training and validation
    loss : STRING
        name of available loss function
        example:
            loss = 'categorical_crossentropy'
    opt : PYTHON DICTIONARY
        with name of optmizer and parameters
        example: 
            opt = {'SGD' : {'lr':0.01, 'momentum':0.0, 'clipvalue':5.}} 
    metrics : LIST
        with names of metrics to be used
        example: 
            metrics = ['accuracy']
    unfreeze : BOOLEAN
        whether or not to set all layers in the model trainable
    patience : INT
        uses Keras EarlyStopping callback function with patience when 
        patience != 0.
    Returns
    -------
    model : TRAINED MODEL
        trained model.
    history : KERAS HISTORY
        training history.
    df : PANDAS DATAFRAME
        when test_dir is not None. A pandas dataframe with the results of model
        classification
    """
  
    # load the model:
    model = load_model(model_path)
   
    # use the provided generator and 
    # do the same thing for both training and validation:    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True) 
    
    valid_generator = datagen.flow_from_directory(
        valid_dir,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) 
    
    if valid_generator.num_classes != train_generator.num_classes:
        print('Warning! Different number of classes in training and validation')

    #################################
    # generators set
    #################################
           
    # compile model with provided parameters:
    opt_name = next(iter(opt))
    if opt_name == 'SGD':
        opt_config = optimizers.SGD(**opt[opt_name])
    elif opt_name == 'Adam':
        opt_config = optimizers.Adam(**opt[opt_name])
    elif opt_name == 'Adadelta':
        opt_config = optimizers.Adadelta(**opt[opt_name])
    elif opt_name == 'Adagrad':
        opt_config = optimizers.Adagrad(**opt[opt_name])
    elif opt_name == 'Adamax':
        opt_config = optimizers.Adamax(**opt[opt_name])
    elif opt_name == 'Ftrl':
        opt_config = optimizers.Ftrl(**opt[opt_name])
    elif opt_name == 'Nadam':
        opt_config = optimizers.Nadam(**opt[opt_name])
    elif opt_name == 'RMSprop':
        opt_config = optimizers.RMSprop(**opt[opt_name])

    if unfreeze:
        for layer in model.layers:
            layer.trainable = True
    
    model.compile(optimizer=opt_config,
                  loss=loss, 
                  metrics=metrics)
 
    print(f'model needs {get_model_memory_usage(batch_size, model)} Gb')
    
    if patience != 0:
        history = model.fit_generator(generator=train_generator,
                                      validation_data=valid_generator,
                                      shuffle=True,
                                      steps_per_epoch=len(train_generator),
                                      validation_steps=len(valid_generator),
                                      epochs=epochs, 
                                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                                  min_delta = 1e-6,
                                                                                  patience=patience)])

    else:
        history = model.fit_generator(generator=train_generator,
                                      validation_data=valid_generator,
                                      shuffle=True,
                                      steps_per_epoch=len(train_generator),
                                      validation_steps=len(valid_generator),
                                      epochs=epochs)
      
    # predict test values accuracy
    if test_dir is not None:
        generator = datagen.flow_from_directory(
                            test_dir,
                            target_size=model.input_shape[1:3],
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False)
    
        pred = model.predict_generator(generator, 
                                        verbose=0, 
                                        steps=len(generator)
    									)
    
         # save results as dataframe
        df = pd.DataFrame(pred, columns=generator.class_indices.keys())
        df['file'] = generator.filenames
        df['true_label'] = df['file'].apply(os.path.dirname).apply(str.lower)
        df['pred_idx'] = np.argmax(df[generator.class_indices.keys()].to_numpy(), axis=1)
        # save as the label (dictionary comprehension because generator.class_indices has the
        # key,values inverted to what we want
        df['pred_label'] = df['pred_idx'].map({value: key for key, value in generator.class_indices.items()}).apply(
            str.lower)
        # save the maximum probability for easier reference:
        df['max_prob'] = np.amax(df[generator.class_indices.keys()].to_numpy(), axis=1)
   
        return model, history, df
    return model, history

