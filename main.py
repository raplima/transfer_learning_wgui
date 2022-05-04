# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:34:37 2020

@author: rafael.lima
uses functions from other modules to fit and evaluate models
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from plotting_functions import plot_history, plot_confusion_matrix
from helper_functions import train_valid_test_split, model_preprocess, get_models_dict
from model_fit import make_model, train_model
from model_evaluate import classify_folders, compute_metrics


if __name__ == "__main__":
  
    # set the seed for repetition purposes 
    np.random.seed(0)
    tf.random.set_seed(0)
    
    data_dir = 'C:/Users/rafael.lima/Desktop/data/simple'
    image_dir = '../images'
    model_dir = '../models'
    
    # for model selection parameters
    models_dict = get_models_dict()
    
    model_pars = [#{'Dropout': {'rate': 0.5}}, 
                  {'Dense':   {'units': 100, 'activation': 'relu'}}, 
                  {'Dense':   {'units': 4, 'activation': 'softmax'}}]
 
    model_name = 'VGG16'
    weights = 'imagenet'
    
    # split data into train, validation, test:
    train_dir, valid_dir, test_dir = train_valid_test_split(data_dir, val_frac=0.1, test_frac=0.15)
    
    
    # using the default graph helps with memory in some cases
    # specially for multiple trainings
    g = tf.Graph()  
    with g.as_default():
        model = make_model(model_name, models_dict[model_name], model_pars, weights)
        # save model:
        model_path = os.path.join(model_dir, f"{model_name}.hdf5")
        model.save(model_path)
        
        # set the generator:
        datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))
        
        #----------------------------------------------------------------------
        # first step: feature extraction / frozen training:
        epochs = 5
        batch_size = 32
        loss = 'categorical_crossentropy'
        opt_1 = {'Adam' : {}} 
        metrics = ['accuracy']
        model, hist, df = train_model(model_path, 
                                         train_dir, valid_dir, test_dir, 
                                         datagen, 
                                         epochs, 
                                         batch_size,
                                         loss, 
                                         opt_1, 
                                         metrics, 
                                         unfreeze = False, 
                                         patience = 5)

        model_path = os.path.join(model_dir, f"{model_name}-feat_extract.hdf5")
        model.save(model_path)
        
        #----------------------------------------------------------------------
        # second step: fine-tuning
        epochs = 3
        opt_2 = {'SGD' : {'lr':1e-4, 'momentum':0.0, 'clipvalue':5.}} 

        model, hist_2, df = train_model(model_path, 
                                         train_dir, valid_dir, test_dir, 
                                         datagen, 
                                         epochs, 
                                         batch_size,
                                         loss, 
                                         opt_2, 
                                         metrics, 
                                         unfreeze = True, 
                                         patience = 3)
        
        for metr in hist_2.history:
            for jj in hist_2.history[metr]:
                hist.history[metr].append(jj)
        
        model_path = os.path.join(model_dir, f"{model_name}-fine_tuned.hdf5")
   
        model.save(model_path)
        
        for dset in [train_dir, valid_dir, test_dir]:
            df_res = classify_folders(dset, model_path, model_name)
            y_true = df_res['true'].to_numpy()
            y_pred = df_res['pred'].to_numpy()
            
            dict_metrics = compute_metrics(y_true, y_pred)
            print(f"dataset: {dset}, accurac: {dict_metrics['accuracy']}")
            
        _ = plot_history(hist, image_dir=image_dir)
        _ = plot_confusion_matrix(dict_metrics['confusion_matrix'],                                   
                                  image_dir=image_dir)