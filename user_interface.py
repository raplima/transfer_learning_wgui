# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:32:03 2020

@author: rafael.lima
user interface for transfer learning
"""

import os

import PySimpleGUI as sg
import json

import pandas as pd # used to save dataframes
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from helper_functions import train_valid_test_split, model_preprocess, get_models_dict
from model_fit import make_model, train_model
from model_evaluate import classify_folders, compute_metrics
from plotting_functions import plot_history, plot_confusion_matrix

class opt_window:

    def __init__(self, this_optimizer_number, opt_pars):
        self.layout = [[sg.T(f'Optimizer {this_optimizer_number}',
                             size=size1,
                             tooltip ='Optimizer is written as a json file below. \n'+ 
                             'Watch out for typos if you choose to write the parameters. \n')],
                       [sg.Multiline(default_text=json.dumps(opt_pars,indent=4), 
                                     size = (100, 10), 
                                     key=f'opt_{optimizer_number}', 
                                     enable_events=True)],

                       [sg.Button('Use optimizer with provided hyperparameters', 
                                  key=f'button_opt_{this_optimizer_number}_use')],
                       [sg.Button('Reset optimizer', 
                                  key=f'button_opt_{this_optimizer_number}_reset')]
                      ]

        self.window = sg.Window(f'Advanced optimizer {this_optimizer_number} parameters', 
                                layout=self.layout)


if __name__ == "__main__":
    
    # GUI theme
    sg.theme('Dark Blue 3')
    
    # models dictionary and list
    models_dict = get_models_dict()
    models_available = [k for k in models_dict]
    # default model:
    base_model = 'VGG16'
    
    # top model hyperparameters (layers)
    model_pars = [{'Dropout': {'rate': 0.5}}, 
                  {'Dense':   {'units': 100, 'activation': 'relu'}}, 
                  {'Dense':   {'units': 2, 'activation': 'softmax'}}]
    # default top model parameters
    simple_model_pars = model_pars.copy()
    
    # default optimizer hyperparameters
    opt_feature_extraction = {'Adam' : {}} 
    opt_fine_tune = {'SGD' : {'lr':1e-4, 'momentum':0.0, 'clipvalue':5.}} 
    
    # keras metrics to be used for evaluation
    metrics = ['accuracy']
    
    # optimizers     
    opt_1 = opt_feature_extraction.copy()
    opt_2 = opt_fine_tune.copy()
    
    # optimizer dictionary for selection
    opt_dict = {'Adam' : {},
                'SGD': {'clipvalue':5.},
                'Adadelta' : {},
                'Adagrad': {},
                'Adamax': {},
                'Ftrl' : {},
                'Nadam' : {},
                'RMSprop' : {'clipvalue':5.},
                }
    
    # training loss
    loss = 'categorical_crossentropy'
    
    # list of number of epochs for combo box
    epochs_choice = [1, 5, 10, 20, 50, 100, 500, 1000]
    # list of batch size choices
    batch_choice = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
    size1 = (25, 1)
    size2 = (30, 1)
    size_model_window = (100, 20)

    # define some folder locations:
    initial_folder = os.getcwd()
    image_dir = os.getcwd()
    model_dir = os.getcwd()
    results_dir = os.getcwd()
    model_path = None
    
    print = sg.Print
    
    #Main window
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    # set layout by tab
    tab1_layout =  [
                    [sg.T('Select data folder: ', size=size1),
                     sg.In(key='data_in'),
                     sg.FolderBrowse(initial_folder=initial_folder, 
                                     tooltip='Folder containing data to be split',
                                     key='data_inbrowse')
                    ],
                    [sg.Frame('Validation Percentage', [[
                            sg.Slider(key='val_per',
                                      range=(1, 99), 
                                      orientation='h',                                
                                      default_value=10, 
                                      tooltip='Percentage of data to be used as validation.')
                    ]]),
                     
                     sg.Frame('Validation Percentage', [[
                             sg.Slider(key='test_per',
                                       range=(0, 99), 
                                       orientation='h',                                
                                       default_value=10,
                                       tooltip='Percentage of data to be used as test.')
                    ]])

                    
                    ],
                    
                    
                    [sg.Text('_' * size1[0], 
                             size=size1,
                             justification='center')],

                    
                    [sg.Button('Split data',
                               size=size1, 
                               key='button_split',
                               tooltip='Split data with parameters provided.'),
                        ]
                                            
                    ]

    tab2_layout = [ [sg.T('Select training data folder: ', size=size1),
                     sg.In(key='train_dir', enable_events=True),
                     sg.FolderBrowse(initial_folder=initial_folder, 
                                     tooltip='Folder containing training data',
                                     key='train_dirbrowse')],
                   
                   [sg.T('Select validation data folder: ', size=size1),
                     sg.In(key='valid_dir', enable_events=True),
                     sg.FolderBrowse(initial_folder=initial_folder, 
                                     tooltip='Folder containing validation data',
                                     key='valid_dirbrowse')],
                   [sg.T('Choose CNN base model: ', size=size1),
                    sg.Combo(models_available, 
                             default_value=base_model, 
                             key='model_name'), 
                    sg.Button('Advanced top model options', 
                              key = 'button_model_params')
                    ],

                   [sg.T('Batch size:',
                         size = size1 ,
                         tooltip='Number of samples that are used in one iteration.\n'+
                         '(iteration over all samples within one set is equivalent to 1 epoch.)'),
                    sg.Combo(batch_choice,
                             key='batch_size',
                             default_value=32, 
                             tooltip='Number of samples - use largest number possible. \n'+
                                     'Limited by hardware (memory).')
                   ],
                   
                   
                   [sg.Frame('Step 1', [[sg.T('Epochs:', 
                                              tooltip='Number of epochs used for the first step of fine-tuning.'),
                                         sg.Combo(epochs_choice, 
                                                  key='epochs_1',
                                                  default_value=50, 
                                                  tooltip='Number of epochs used for the first step of fine-tuning.')],
                                        [
                                        sg.Checkbox('Ealy stopping', 
                                                    default=True, 
                                                    key='early_1', 
                                                    tooltip='Stop training when validation loss has stopped improving in step 1.')],
                                        [
                              
                                        sg.Button('Advanced optimizer options', 
                                                  key='button_optimizer1',
                                                  tooltip='Click to provide advanced options for optimizer used in the second step of fine-tuning.'),
                                        ]], size=size1),
                                                  
                    sg.Frame('Step 2', [[sg.T('Epochs:', 
                                              tooltip='Number of epochs used for the second step of fine-tuning.'),
                                         sg.Combo(epochs_choice, 
                                                  key='epochs_2',
                                                  default_value=50, 
                                                  tooltip='Number of epochs used for the second step of fine-tuning.')],
                                        [
                                        sg.Checkbox('Ealy stopping', 
                                                    default=True, 
                                                    key='early_2', 
                                                    tooltip='Stop training when validation loss has stopped improving in step 2.')],
                                        [
                              
                                        sg.Button('Advanced optimizer options', 
                                                  key='button_optimizer2',
                                                  tooltip='Click to provide advanced options for optimizer used in the second step of fine-tuning.'),
                                        ]], size=size1)
                                                  
                                                  
                    ],
                      
                   
                   [sg.Text('_' * size1[0], 
                            size=size1,
                            justification='center')],

                    [sg.Button('Start training', 
                              size = size1 ,
                              key='button_start_training',
                              tooltip='Start training the model with provided parameters.')
                    ]]
                    
                              
                   
    tab3_layout = [  [sg.T('Select model: ', 
                           size=size1, 
                           tooltip='Select the trained CNN model to be evaluated.'),
                      sg.In(key='model_path', enable_events=True),
                      sg.FileBrowse(initial_folder=initial_folder, 
                                    tooltip='Model to be evaluated.',
                                    key='model_filebrowse')],
                    
                   [sg.T('Select CNN base model: ', 
                         size=size1),
                    sg.Combo(models_available, 
                             default_value=base_model, 
                             enable_events=True,
                             key='model_name_eval')
                    ],

                    [sg.T('Select test data folder: ', 
                          size=size1, 
                          tooltip='Select the folder with images to be evaluated.'),
                     sg.In(key='test_dir', enable_events=True),
                     sg.FolderBrowse(initial_folder=initial_folder, 
                                     tooltip='Folder containing training data',
                                     key='test_dirbrowse')],
                    
                    [sg.Frame(layout=[
                            [sg.Checkbox('Plot confusion matrix', 
                                         size=size1, 
                                         default=True, 
                                         key='plot_confusion_matrix', 
                                         tooltip='Save a figure showing the confusion matrix.'),  
                             sg.Checkbox('Save confusion matrix', 
                                         size=size1, 
                                         key='save_confusion_matrix', 
                                         tooltip='Save a csv file with the confusion matrix information.')],

                            [sg.Checkbox('Accuracy', 
                                         size=size1, 
                                         default=True, 
                                         key='want_accuracy', 
                                         tooltip='Include accuracy in the metrics.'),  
                             sg.Checkbox('Balanced accuracy', 
                                         size=size1, 
                                         key='want_balanced_accuracy', 
                                         tooltip='Include balanced accuracy in the metrics.')],

                            [sg.Checkbox('Kappa', 
                                         size=size1, 
                                         default=True, 
                                         key='want_kappa', 
                                         tooltip="Include Cohen's kappa in the metrics."),  
                             sg.Checkbox('Classification report', 
                                         size=size1, 
                                         key='want_classification_report',
                                         tooltip='Include a classification report containing class-by-class metrics.')],
        

                    ], 
                     title='Options', relief=sg.RELIEF_SUNKEN, tooltip='Select options for model evaluation')],
                    
                    [sg.Text('_' * size1[0], 
                             size=size1,
                             justification='center')],
                                     
                    [sg.Button('Evaluate', 
                              size = size1 ,
                              key = 'button_evaluate',
                              tooltip='Start training the model with provided parameters.')
                    ]
                   
                   ]
    
    layout = [[sg.TabGroup([[sg.Tab('Data Split', tab1_layout), 
                             sg.Tab('Fit CNN model', tab2_layout),
                             sg.Tab('Evaluate CNN model', tab3_layout)]])]]
    
    window = sg.Window('CNN transfer learning for image classification', 
                       layout, auto_size_text=True)
    
    #End main window
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    window_model_active = False
    
    window_opt1_active = False
    window_opt2_active = False
    
    while True:
        event, values = window.read()
        if event is None or event == 'Exit':
            break
        if event == 'button_split':
            # split data into train, validation, test:
            try:
                res = train_valid_test_split(values['data_in'], 
                                             val_frac=values['val_per']/100., 
                                             test_frac=values['test_per']/100.)
                window.Element('train_dir').Update(res[0])
                window.Element('valid_dir').Update(res[1])
                window.Element('test_dir').Update(res[2])
                
                values["train_dir"] = res[0]
                values["valid_dir"] = res[1]
                values["test_dir"] = res[2]
                
                model_pars.pop()
                model_pars.append({'Dense':   {'units': f'{len(os.listdir(values["train_dir"]))}', 'activation': 'softmax'}})

                sg.Popup('Data split complete.')

            except Exception as e:
                print(e)
                sg.PopupError('Please check input parameters.')
                
        # Tab 2 buttons:
        # ---------------------------------------------------------------------
        if event == 'button_start_training':
            # use provided parameters and start training model
            try:
                # check if top model has the correct number of neurons for the last layer:
                try:
                    if int(model_pars[-1]['Dense']['units']) != len([d for d in Path(values['train_dir']).iterdir() if d.is_dir()]):
                        sg.Popup('Warning! Number of dense layers in last layer automatically modified to match number of subfolders in training folder')
                        model_pars.pop()
                        model_pars.append({'Dense':   {'units': f'{len(os.listdir(values["train_dir"]))}', 'activation': 'softmax'}})
                except Exception as e:
                    print(e)
                    sg.PopupError('Please check model parameters.')                        

                # using the default graph helps with memory in some cases
                # specially for multiple trainings
                g = tf.Graph()  
                with g.as_default():
                # assemble model with provided parameters                
                    model = make_model(values['model_name'], 
                                       models_dict[values['model_name']], 
                                       model_pars, 
                                       'imagenet')
                    # save model:
                    model_path = os.path.join(model_dir, f"{values['model_name']}.hdf5")
                    model.save(model_path)
    
                    # set the generator:
                    datagen = ImageDataGenerator(preprocessing_function=model_preprocess(values['model_name']))
                    
                    model, hist, _ = train_model(model_path,
                                                  values['train_dir'],
                                                  values['valid_dir'], 
                                                  values['valid_dir'], 
                                                  datagen, 
                                                  values['epochs_1'],
                                                  values['batch_size'],
                                                  loss, 
                                                  opt_1,
                                                  metrics, 
                                                  unfreeze = False, 
                                                  patience = 5 if values['early_1'] else 0)
                    model_path = os.path.join(model_dir, f"{values['model_name']}-feat_extract.hdf5")
                    model.save(model_path)
                
                    #----------------------------------------------------------------------
                    # second step: fine-tuning
                    model, hist_2, _ = train_model(model_path,
                                                    values['train_dir'],
                                                    values['valid_dir'], 
                                                    values['valid_dir'], 
                                                    datagen, 
                                                    values['epochs_2'],
                                                    values['batch_size'],
                                                    loss, 
                                                    opt_2,
                                                    metrics, 
                                                    unfreeze = True, 
                                                    patience = 5 if values['early_2'] else 0)
                    
                    for metr in hist_2.history:
                        for jj in hist_2.history[metr]:
                            hist.history[metr].append(jj)
                    
                    model_path = os.path.join(model_dir, f"{values['model_name']}-fine_tuned.hdf5")
                    model.save(model_path)

                window.Element('model_path').Update(model_path)
                window.Element('model_name_eval').Update(values['model_name'])
                
                _ = plot_history(hist, image_dir=image_dir)
                del g                
                sg.Popup('Training complete.')
                
            except Exception as e:
                print(e)
                sg.PopupError('Please check input parameters for tab 2. ')

        # Tab 2 model params
        # ---------------------------------------------------------------------
                
        if not window_model_active and event == 'button_model_params':
            window_model_active = True
            # open new window for model parameters:
            layout_model = [[sg.T('Top model', size=size1, 
                                  tooltip ='Top model is written as a json file below. \n'+
                                  'You can use buttons on the bottom of the page to add and remove layers. \n'+
                                  'You can also edit the area below to modify the layers. \n'+
                                  'Watch out for typos if you choose to write the parameters. \n'+
                                  'All layer parameters for Dense and Dropout in Keras should be available below.')],
                            [sg.Multiline(default_text=json.dumps(model_pars,indent=4), 
                                          key='top_model',
                                          size = size_model_window,
                                          enable_events=True),],
                            [sg.Button('Remove last layer', 
                                       key='pop_last', 
                                       tooltip = 'Remove the last layer of the top model.'), 
                             sg.Button('Add dropout layer',
                                       key='add_dropout',
                                       tooltip='Add a new dropout layer on the top model.'),
                             sg.Button('Add dense layer',
                                       key='add_dense',
                                       tooltip='Add a new dense layer on the top model. \n'+
                                               'Make sure last layer is a dense layer. \n'+
                                               'Make sure the number of neurons of last layer match the number of classes'),
                             sg.Button('Reset to initial model',
                                       key='reset_top_model',
                                       tooltip='Resets to a simple top model. \n'+
                                               'Make sure the number of neurons of last layer match the number of classes.'),
                             sg.Button('Apply',
                                       key='use_this_model',
                                       tooltip='Save displayed top model.'),

                                       ]]

            window_model = sg.Window('Advanced top model parameters', 
                                     layout=layout_model)
            while window_model_active:
                ev_model, val_model = window_model.read()
                if ev_model is None or ev_model == 'Exit':
                    window_model_active = False
                    window_model.close()
                    del window_model
                    
                if ev_model == 'pop_last':
                    try:
                        model_pars.pop()
                    except:
                        print('error')
                    window_model.Element('top_model').Update(json.dumps(model_pars,indent=4))

                if ev_model == 'add_dropout':
                    try:
                        model_pars.append({'Dropout': {'rate': 0.5}})
                    except:
                        print('error')
                    window_model.Element('top_model').Update(json.dumps(model_pars,indent=4))
                    
                if ev_model == 'add_dense':
                    try:
                        model_pars.append({'Dense':   {'units': 100, 'activation': 'relu'}})
                    except:
                        print('error')
                    window_model.Element('top_model').Update(json.dumps(model_pars,indent=4))                        
                    
                if ev_model == 'reset_top_model':
                    model_pars = simple_model_pars.copy()
                    window_model.Element('top_model').Update(json.dumps(model_pars,indent=4))     

                if ev_model == 'use_this_model':
                    try:
                        model_pars = json.loads(val_model[f'top_model'])
                    except Exception as e: 
                        print(e)
                        sg.PopupError('Something went wrong.')
        
        
        # Tab 2 optimizer 1
        # ---------------------------------------------------------------------        
        if not window_opt1_active and event == 'button_optimizer1':
            window_opt1_active = True
            optimizer_number = 1 
            # open new window for optimizer parameters:
            window_opt1 = opt_window(optimizer_number, opt_1).window
            while window_opt1_active:
                ev_, val_ = window_opt1.read()
                if ev_ is None or ev_ == 'Exit':
                    window_opt1_active = False
                    window_opt1.close()
                    del window_opt1
                    
                if ev_ == f'button_opt_{optimizer_number}_use':
                    try:
                        opt_1 = json.loads(val_[f'opt_{optimizer_number}'])
                    except Exception as e:
                        print(e)
                        sg.PopupError('Please check input parameters.')

                if ev_ == f'button_opt_{optimizer_number}_reset':
                    try:
                        opt_1 = opt_feature_extraction.copy()
                        window_opt1.Element(f'opt_{optimizer_number}').Update(json.dumps(opt_1,indent=4))
                    except:
                        sg.PopupError('Something went wrong.')
                      
        # Tab 2 optimizer 2
        # ---------------------------------------------------------------------        
        if not window_opt2_active and event == 'button_optimizer2':
            window_opt2_active = True
            optimizer_number = 2 
            # open new window for optimizer parameters:
            window_opt2 = opt_window(optimizer_number, opt_2).window
            while window_opt2_active:
                ev_, val_ = window_opt2.read()
                if ev_ is None or ev_ == 'Exit':
                    window_opt2_active = False
                    window_opt2.close()
                    del window_opt2
                    
                if ev_ == f'button_opt_{optimizer_number}_use':
                    try:
                        opt_2 = json.loads(val_[f'opt_{optimizer_number}'])
                    except Exception as e:
                        print(e)
                        sg.PopupError('Please check input parameters.')

                if ev_ == f'button_opt_{optimizer_number}_reset':
                    try:
                        opt_2 = opt_fine_tune.copy()
                        window_opt2.Element(f'opt_{optimizer_number}').Update(json.dumps(opt_2,indent=4))
                    except Exception as e:
                        print(e)
                        sg.PopupError('Something went wrong.')
        
        # Tab 3 button:
        # ---------------------------------------------------------------------
        if event == 'button_evaluate':
            try:
                g = tf.Graph()  
                with g.as_default():
                    df_res = classify_folders(values['test_dir'] ,
                                              values['model_path'], 
                                              values['model_name_eval'])
                    
                    df_res.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
                    y_true = df_res['true'].to_numpy()
                    y_pred = df_res['pred'].to_numpy()

                dict_metrics = compute_metrics(y_true, y_pred, 
                                               return_acc=values['want_accuracy'], 
                                               return_balanced_acc=values['want_balanced_accuracy'], 
                                               return_kappa=values['want_kappa'], 
                                               return_classification_report=values['want_classification_report'],
                                               return_confusionmatrix=True if values['plot_confusion_matrix'] or values['save_confusion_matrix'] else False)
                if values['plot_confusion_matrix']:
                    _ = plot_confusion_matrix(dict_metrics['confusion_matrix'], 
                                              image_dir=image_dir)
                if values['save_confusion_matrix']:
                    dict_metrics['confusion_matrix'].to_csv(os.path.join(results_dir, 'confusion_matrix.csv'), index=False)
                
                with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
                    for k in dict_metrics:
                        if k != 'confusion_matrix':
                            f.write(f'{k}:\n')
                            f.write(f'{dict_metrics[k]}')
                            f.write('\n\n')
                
                sg.Popup('Evaluation complete.\nCheck saved files.')
            except Exception as e: 
                print(e)
                sg.PopupError('Something went wrong.')
    
    window.close()

