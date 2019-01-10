# Rafael Pires de Lima
# December 2018
# GUI

import os
import pickle
import random
import shutil
import tkinter as tk

import matplotlib
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
from keras import applications, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use('TkAgg')
style.use("seaborn")


validation_per = 10
test_per = 1

# hyperparameters
batch_size = 16

# folders management:
bottleneck_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'bnecks' + os.sep
model_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'models' + os.sep


"""
"Processing" functions:
"""
def model_app(arch, input_tensor):
    """Loads the appropriate convolutional neural network (CNN) model
      Args:
        arch: String key for model to be loaded.
        input_tensor: Keras tensor to use as image input for the model.
      Returns:
        model: The specified Keras Model instance with ImageNet weights loaded and without the top classification layer.
      """
    # function that loads the appropriate model
    if arch == 'Xception':
        model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('Xception loaded')
    elif arch == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('VGG16 loaded')
    elif arch == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('VGG19 loaded')
    elif arch == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('ResNet50 loaded')
    elif arch == 'InceptionV3':
        model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('InceptionV3 loaded')
    elif arch == 'InceptionResNetV2':
        model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('InceptionResNetV2 loaded')
    elif arch == 'MobileNet':
        model = applications.MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False,
                                       input_tensor=input_tensor)
        print('MobileNet loaded')
    elif arch == 'DenseNet121':
        model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('DenseNet121 loaded')
    elif arch == 'NASNetLarge':
        model = applications.NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('NASNetLarge loaded')
    elif arch == 'MobileNetV2':
        model = applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False,
                                         input_tensor=input_tensor)
        print('MobileNetV2 loaded')
    else:
        print('Invalid model selected')
        model = False

    return model

def train_val_test(path, val_p=0.1, test_p=0.1):
    """Splits a single root folder with subfolderss (classes) containing images of different classes
    into train/validation/test sets.
      Args:
        path: String path to a root folder containing subfolders of images.
        val_p: Float proportion of the images to be reserved for validation.
        test_p: Float proportion of the images to be reserved for test.
      Returns:
        No returns.
      """
    # parameters
    # val_p = percentage (proportion) of files to be captured for the validation set
    # test_p = percentage (proportion) of files to be captured for the test set
    # path = picture location
    # for reproducibility
    random.seed(1234)
    # print(random.randint(1, 10000))  # 7221

    print("Splitting data in " + path + " into training/validation/test")

    # uses the file path and walks in folders to select training and test data
    lst = os.listdir(path)
    # assume elements with "." are files and remove from list
    folders = [item for item in lst if "." not in item]

    # create folder to save training/validation/test data:
    path_train = os.path.dirname(path) + '/' + path.split("/")[-1] + '_train'
    if os.path.exists(path_train):
        shutil.rmtree(path_train, ignore_errors=True)
    os.makedirs(path_train)

    path_valid = os.path.dirname(path) + '/' + path.split("/")[-1] + '_validation'
    if os.path.exists(path_valid):
        shutil.rmtree(path_valid, ignore_errors=True)
    os.makedirs(path_valid)

    if test_p > 0:
        path_test = os.path.dirname(path) + '/' + path.split("/")[-1] + '_test'
        if os.path.exists(path_test):
            shutil.rmtree(path_test, ignore_errors=True)
        os.makedirs(path_test)

    # for each one of the folders
    for this_folder in folders:
        print("Current folder: " + this_folder)

        # create folder to save cropped image:
        if os.path.exists(path_train + '/' + this_folder):
            shutil.rmtree(path_train + '/' + this_folder, ignore_errors=True)
        os.makedirs(path_train + '/' + this_folder)

        if os.path.exists(path_valid + '/' + this_folder):
            shutil.rmtree(path_valid + '/' + this_folder, ignore_errors=True)
        os.makedirs(path_valid + '/' + this_folder)

        if test_p > 0:
            if os.path.exists(path_test + '/' + this_folder):
                shutil.rmtree(path_test + '/' + this_folder, ignore_errors=True)
            os.makedirs(path_test + '/' + this_folder)

        # get pictures in this folder:
        lst = os.listdir(path + "/" + this_folder)

        # separate training and test data:
        # get pictures in this folder:
        lst = os.listdir(path + "/" + this_folder)

        if (len(lst) < 3):
            print("      Not enough data for automatic separation")
        else:
            # shuffle the indices:
            np.random.shuffle(lst)

            # number of test images:
            n_valid = np.int(np.round(len(lst) * val_p))
            n_test = np.int(np.round(len(lst) * test_p))
            if n_valid == 0:
                print("Small amount of data, only one sample forcefully selected to be part of validation data")
                n_valid = 1;
            if test_p > 0 and n_test == 0:
                print("Small amount of data, only one sample forcefully selected to be part of test data")
                n_test = 1;

            # copy all pictures to appropriate folder
            for t in range(0, n_test):
                shutil.copy2(path + '/' + this_folder + '/' + lst[t], path_test + '/' + this_folder)

            for t in range(n_test, n_test + n_valid):
                shutil.copy2(path + '/' + this_folder + '/' + lst[t], path_valid + '/' + this_folder)
                # print("copied " + lst[t] + " to validation folder");

            for t in range(n_test + n_valid, len(lst)):
                shutil.copy2(path + '/' + this_folder + '/' + lst[t], path_train + '/' + this_folder)
                # print("copied " + lst[t] + " to train folder");


def save_bottleneck_features(train_data_dir, validation_data_dir, bottleneck_name, img_height, img_width, arch):
    """Saves the bottlenecks of validation and train data.
      Args:
        train_data_dir: String path to a folder containing subfolders of images (training set).
        validation_data_dir: String path to a folder containing subfolders of images (validation set).
        bottleneck_name: String used as main element of bottlenecks files.
        img_height: Integer, image height.
        img_width: Integer, image width.
        arch: String that defines the CNN model to be used.
      Returns:
        No returns. Saves bottlenecks using bottleneck_name and bottleneck_dir
      """
    # Saves the bottlenecks of validation and train data.
    # Input is path to train_data_dir and validation_data_dir (directories with the images)
    # bottleneck_name is the name to be used for saving
    # bottlenck_dir is defined outside of this function
    # arch is the architecture to be used
    global bottleneck_dir
    datagen = ImageDataGenerator(rescale=1. / 255)

    # check to see if runs/botteneck path exists
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    # build the network
    model = model_app(arch, Input(shape=(img_height, img_width, 3)))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, generator.n // batch_size, verbose=1)

    # save a tuple of bottlenecks and the corresponding label
    np.save(open(bottleneck_dir + bottleneck_name + '_train.npy', 'wb'),
            bottleneck_features_train)
    np.save(open(bottleneck_dir + bottleneck_name + '_train_labels.npy', 'wb'),
            generator.classes[0:bottleneck_features_train.shape[0]])

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, generator.n // batch_size, verbose=1)

    # save a tuple of bottlenecks and the corresponding label
    np.save(open(bottleneck_dir + bottleneck_name + '_val.npy', 'wb'),
            bottleneck_features_validation)

    np.save(open(bottleneck_dir + bottleneck_name + '_val_labels.npy', 'wb'),
            generator.classes[0:bottleneck_features_validation.shape[0]])

    # finally, save a "dictionary" as the labels are numeric and eventually we want to know the original string label:
    with open(bottleneck_dir + bottleneck_name + '_dict_l', 'wb') as fp:
        pickle.dump(sorted(os.listdir(train_data_dir)), fp)


def train_top_model(bottleneck_name, model_name, arch, img_height, img_width, epochs, opt):
    """Trains the new classification layer generating the new classfication model dependent on the classes we are using.
      Args:
        bottleneck_name: String used as main element of bottlenecks files.
        model_name: String, name of the model to be saved.
        arch: String that defines the CNN model to be used.
        img_height: Integer, image height.
        img_width: Integer, image width.
        epochs: Integer, the number of epochs (iterations on complete training set) to be performed
        opt: String, optimizer to be used.
      Returns:
        No returns. Trains and saves the model. Opens a tkinter window with training history
      """

    train_data = np.load(open(bottleneck_dir + bottleneck_name + '_train.npy', 'rb'))
    train_labels = np.load(open(bottleneck_dir + bottleneck_name + '_train_labels.npy', 'rb')).reshape(-1)

    validation_data = np.load(open(bottleneck_dir + bottleneck_name + '_val.npy', 'rb'))
    validation_labels = np.load(open(bottleneck_dir + bottleneck_name + '_val_labels.npy', 'rb')).reshape(-1)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    #top_model.add(Dense(512, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5)) # dropout helps with overfitting
    top_model.add(Dense(len(np.unique(validation_labels)), activation='softmax'))

    top_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=1),
                 ModelCheckpoint(filepath=model_dir+'tempbm.h5', monitor='val_acc', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001, verbose=1)]

    history = top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              shuffle=True,
              callbacks=callbacks)

    # reload best model:
    top_model = load_model(model_dir+'tempbm.h5')
    score = top_model.evaluate(validation_data, validation_labels, verbose=0)
    print('{:22} {:.2f}'.format('Validation loss:', score[0]))
    print('{:22} {:.2f}'.format('Validation accuracy:', score[1]))
    print('')

    # save the entire model:
    # build the network
    base_model = model_app(arch, Input(shape=(img_height, img_width, 3)))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    if opt == 'RMSprop':
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    if opt == 'SGD':
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # check to see if runs/model path exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(model_dir + model_name + '.hdf5')
    # also save the dictionary label associated with this file for later testing
    shutil.copy2(bottleneck_dir + bottleneck_name + '_dict_l', model_dir + model_name + '_dict_l')
    # delete temporary model file:
    os.remove(model_dir+'tempbm.h5')

    print('Transfer learning complete.')

    # plotting the metrics
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax[0].plot(range(1, len(history.history['acc'])+1), history.history['acc'])
    ax[0].plot(range(1, len(history.history['acc'])+1), history.history['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='lower right')

    ax[1].plot(range(1, len(history.history['acc'])+1), history.history['loss'])
    ax[1].plot(range(1, len(history.history['acc'])+1), history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper right')

    # set up figure
    fig.set_size_inches(w=5, h=7)

    # put the figure in a tkinter window:
    # initialize the window
    root = tk.Tk()
    root.config(background='white')

    graph = FigureCanvasTkAgg(fig, master=root)
    graph.get_tk_widget().pack(side="top", fill='both', expand=True)
    graph.draw()
    graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(graph, root)
    toolbar.update()

    graph._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()


def label_one(path_img, path_model):
    """Labels (classifies) a single image based on a retrained CNN model.
      Args:
        path_img: String path to a single image.
        path_model: String path to the model to be used for classification.
      Returns:
        No returns. Performs the classification and opens a tkinter window with the image and probabilities assigned.
      """
    # load the model:
    model = load_model(path_model)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    # load the image
    img = image.load_img(path_img, target_size=(img_height, img_width))

    # save as array and rescale
    x = image.img_to_array(img) * 1. / 255

    # predict the value
    pred = model.predict(x.reshape(1, img_height, img_width, 3))
    return pred

def label_folder(path_folder, path_model):
    """Labels (classifies) a folder containing subfloders based on a retrained CNN model.
      Args:
        path_folder: String path to a folder containing subfolders of images.
        path_model: String path to the model to be used for classification.
      Returns:
        List: a numpy array with predictions (pred) and the file names of the images classified (generator.filenames)
      """
    # load the model:
    model = load_model(path_model)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    datagen = ImageDataGenerator(rescale=1. / 255)

    # try flow from directory:
    generator = datagen.flow_from_directory(
        path_folder,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    generator = datagen.flow_from_directory(
        path_folder,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    if len(generator)>0:
        # if data file is structured as path_folder/classes, we can use the generator:
        pred = model.predict_generator(generator, steps=len(generator), verbose=1)
    else:
        # the path_folder contains all the images to be classified
        # TODO: if problems arise
        pass

    return [pred, generator.filenames]