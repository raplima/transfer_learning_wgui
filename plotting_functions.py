# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:51:21 2020

@author: rafael.lima
plotting functions 
"""
import os
from itertools import product
import numpy as np
from matplotlib import pyplot as plt

def plot_history(history, 
                 figsize = (7.5, 4.5),
                 show_fig = True, 
                 image_dir = None, 
                 format_ext = 'pdf',
                 dpi = None):
    """
    plots training loss and accuracy history

    Parameters
    ----------
    history : KERAS HISTORY 
        Keras training history
    figsize : TUPLE, optional
        Figure size in inches. The default is (7,5, 4,5).
    show_fig : BOOLEAN, optional
        True to show figure with fig.show(). . The default is True.
    image_dir : STRING, OS.PATH, optional
        Path to save the figure. The default is None.
    format_ext : STRING, optional
        Figure format extension. The default is '.pdf'.
    dpi : INT, optional
        Dots per inch when figure is saved in 'png', 'jpg', etc. The default is None.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(nrows=1, ncols=2, 
                           figsize=figsize, 
                           sharex=True, 
                           facecolor='w')
    
    ax[0].plot(np.arange(1, len(history.history['loss'])+1), 
               history.history['loss'], 
               marker = 'o',
               label = 'Training')
    ax[0].plot(np.arange(1, len(history.history['loss'])+1), 
               history.history['val_loss'], 
               marker = 'o',
               label = 'Validation')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim((-0.05,1.05*np.max(history.history['val_loss'])))
    
    ax[1].plot(np.arange(1, len(history.history['accuracy'])+1), 
               history.history['accuracy'], 
               marker = 'o',
               label = 'Training')
    ax[1].plot(np.arange(1, len(history.history['loss'])+1), 
               history.history['val_accuracy'], 
               marker = 'o',
               label = 'Validation')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim((-0.05,1.05))
    
    ax[1].legend(loc='lower right')
    
    fig.tight_layout()
   
    if show_fig:
        fig.show()
 
    if image_dir:
        if dpi:
            fig.savefig(os.path.join(image_dir, f'training_val_loss_acc.{format_ext}'), dpi=dpi)
        else:
            fig.savefig(os.path.join(image_dir, f'training_val_loss_acc.{format_ext}'))
       
    return fig

def plot_confusion_matrix(df, cmap='Greens', 
                          include_values=True,
                          values_format=None,
                          xticks_rotation = 45,
                          figsize = (7.5, 4.5),
                          show_fig = True, 
                          image_dir = None, 
                          format_ext = 'pdf',
                          dpi = None):

    """
    plots confusion matrix
    modified from https://github.com/scikit-learn/scikit-learn/blob/00fe3d6944f91d52b24d0f59cc9a4dd83be99bcf/sklearn/metrics/_plot/confusion_matrix.py#L119

    Parameters
    ----------
    df : pandas dataframe
        confusion matrix
    cmap : str or matplotlib Colormap, default='Greens'
            Colormap recognized by matplotlib.        
    include_values : boolean
        Include values if true
    figsize : TUPLE, optional
        Figure size in inches. The default is (7,5, 4,5).
    show_fig : BOOLEAN, optional
        True to show figure with fig.show(). . The default is True.
    image_dir : STRING, OS.PATH, optional
        Path to save the figure. The default is None.
    format_ext : STRING, optional
        Figure format extension. The default is '.pdf'.
    dpi : INT, optional
        Dots per inch when figure is saved in 'png', 'jpg', etc. The default is None.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    display_labels = np.unique(list(df))
    n_classes = len(display_labels)
    cm = df.to_numpy()

    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(nrows=1, ncols=1, 
                           figsize=figsize, 
                           sharex=True, 
                           facecolor='w')
    ax.grid(False)
   
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    lab_text_ = None

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    
    if include_values:
        lab_text_ = np.empty_like(df, dtype=object)
        if values_format is None:
            values_format = 'd'

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(len(cm)), range(len(cm))):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            lab_text_[i, j] = ax.text(j, i,
                                       format(cm[i, j], values_format),
                                       ha="center", va="center",
                                       color=color)
    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           xlabel="True label",
           ylabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)    
    fig.tight_layout()
   
    if show_fig:
        fig.show()
 
    if image_dir:
        if dpi:
            fig.savefig(os.path.join(image_dir, f'confusion_matrix.{format_ext}'), dpi=dpi)
        else:
            fig.savefig(os.path.join(image_dir, f'confusion_matrix.{format_ext}'))
            
    return fig
