# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:42:15 2020

@author: rafael.lima
"""


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import metrics

from helper_functions import model_preprocess


def classify_folders(data_dir, model_path, model_name = None):
    """
    Labels (classifies) a folder containing subfolders using a CNN model

    Parameters
    ----------
    data_dir : STRING (os.path)
        Path to a folder containing sufolders of images.
    model_path : Keras/Tensorflow model
        Path to the model to be used for classification.
    model_name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : pandas dataframe
        A dataframe containing the assigned probabilities, 
        as well as the true and predicted labels.
    """
    # load the model:
    model = load_model(model_path)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]
    
    if model_name is not None:
        datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))
    else:
        print("No preprocessing, images will be rescaled 1/255.")
        datagen = ImageDataGenerator(rescale=1/255.)

    # flow from directory:
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    if len(generator) > 0:
        # if data file is structured as path_folder/classes, we can use the generator:
        pred = model.predict_generator(generator, steps=len(generator), verbose=1)
        # put it into a pandas dataframe:
        df = pd.DataFrame(generator.filenames, columns=['file'])
        for k in generator.class_indices:
            df[k] = pred[:, generator.class_indices[k]]

        # for this test, we know the real label:
        inv_map = {v: k for k, v in generator.class_indices.items()}
        true_label = np.array([inv_map[ind] for ind in generator.classes])
        df['true'] = true_label
        df['pred'] = df[[k for k in generator.class_indices]].idxmax(axis=1)
        # save the maximum probability assigned
        df['max_pred'] = df[[k for k in generator.class_indices]].max(axis=1)
        
        return df

    else:
        # the path_folder contains all the images to be classified
        print('Please very folder structure, program expects nested folders as input')
        # TODO: if problems arise
        pass

def compute_metrics(y_true, y_pred, 
                    return_acc=True, 
                    return_balanced_acc=True, 
                    return_kappa=True, 
                    return_classification_report=True,
                    return_confusionmatrix=True):
    metrics_dict = {}
    if return_acc:
        metrics_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if return_balanced_acc:
        metrics_dict['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true, y_pred)
    if return_kappa:
        metrics_dict['cohen_kappa'] = metrics.cohen_kappa_score(y_true, y_pred)
    if return_classification_report:
        metrics_dict['classification_report'] = metrics.classification_report(y_true, y_pred)
    if return_confusionmatrix:
        cm = metrics.confusion_matrix(y_true, y_pred)
        # transpose matrix to match R's confusion matrix
        # with Predicted in the y axis and True in the x axis
        cm = np.transpose(cm)
        df = pd.DataFrame()
        labels = np.unique(np.concatenate((y_true, y_pred)))
        # rows
        for i, row_label in enumerate(labels):
            rowdata={}
            # columns
            for j, col_label in enumerate(labels): 
                rowdata[col_label]=cm[i,j]
            df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
        metrics_dict['confusion_matrix'] = df

    return metrics_dict
