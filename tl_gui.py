# Rafael Pires de Lima
# January 2019
# GUI

#impor the processing script:
import tl_processing as tl

import os
import pickle
import random
import shutil
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

import matplotlib
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use('TkAgg')
style.use("seaborn")

LARGE_FONT = ('Verdana', 12)
NORMAL_FONT = ('Verdana', 10)
NORMAL_FONT_BOLD = ('Verdana', 10, 'bold')
SMALL_FONT = ('Verdana', 8)

validation_per = 10
test_per = 1

# hyperparameters
batch_size = 16

# folders management:
bottleneck_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'bnecks' + os.sep
model_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'models' + os.sep


"""
Graphic interface functions
"""

def popupmsg(msg):
    """Pops up a tkinter window with a message
      Args:
        msg: String, the message to be displayed.
      Returns:
        No returns. Opens a tkinter window with a message.
    """
    popup = tk.Tk()

    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORMAL_FONT)
    label.pack(side='top', fill='x', pady=10)
    B1 = ttk.Button(popup, text='Okay', command=popup.destroy)
    B1.pack()
    popup.mainloop()

def popupgraph(fig):
    """Pops up a tkinter window with a matplotlib figure
          Args:
            fig: A matplotlib figure String, the message to be displayed.
          Returns:
            No returns. Opens a tkinter window with the matplotlib figure.
    """
    # receives a matplotlib image to plot in a new window.
    popup = tk.Tk()

    popup.wm_title("Graph")

    canvas = FigureCanvasTkAgg(fig, popup)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, popup)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    B1 = ttk.Button(popup, text='close', command=popup.destroy)
    B1.pack()
    popup.mainloop()


def popupmultigraph(df, path, model_labels, i=0):
    """Uses a pandas data frame to pop up a tkinter window with a matplotlib figure and next/previous controls
        Args:
            df: A pandas data frame containing the results of classification performed by a CNN model
            path: String path to the folder containing subfolders of the classified images.
            model_labels: A python list with the real name of the classes
            i: Integer, the row to be used for plotting.
                fig: A matplotlib figure String, the message to be displayed.
        Returns:
            No returns. Plots results in the popup window with options to check next and previous images.
    """

    # use i as counter
    # i = 0
    global row
    row = i

    def qc_win():
        # initialize the window
        root = tk.Tk()
        root.config(background='white')

        # set up figure
        fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        ax[0].axis("off")

        ax[1].set_xlabel("Probability")

        fig.set_size_inches(w=14, h=6)

        graph = FigureCanvasTkAgg(fig, master=root)
        graph.get_tk_widget().pack(side="top", fill='both', expand=True)

        # get the class position for later plotting:
        x_pos = [elem for elem, _ in enumerate(model_labels)]

        def get_values(pos):
            img = path + os.sep + df.iloc[[pos]].values[0][-1]
            data = df.iloc[[pos]].values[0][0:-1].reshape((1, len(df.columns) - 1))

            return [img, data]

        def plotter(move_to):
            global row

            if move_to == 'prev' and row > 0:
                row = row - 1
            if move_to == 'next' and row < len(df)-1:
                row = row + 1

            ax[0].cla()
            ax[1].cla()

            [img_p, data] = get_values(row)

            # read image for plotting:
            img = mpimg.imread(img_p)
            ax[0].axis("off")
            ax[0].set_title(df.iloc[[row]].values[0][-1])
            ax[0].imshow(img)

            ax[1].barh(x_pos, data[0][:], color='grey')
            ax[1].set_xlabel("Probability", fontsize=16)
            ax[1].tick_params(labelsize=14)
            ax[1].set_xlim(0.0, 1.0)
            ax[1].yaxis.grid(False)
            ax[1].set_yticks(x_pos)
            ax[1].set_yticklabels('')

            for y, lab in enumerate(model_labels):
                ax[1].text(0, y, lab.replace('_',' '), verticalalignment='center', fontsize=18)

            graph.draw()

        # plot the first image
        plotter('prev')

        button1 = ttk.Button(root, text="Previous", command=lambda: plotter('prev'))
        button1.pack()

        button2 = ttk.Button(root, text="Next", command=lambda: plotter('next'))
        button2.pack()

        root.mainloop()

    qc_win()

def make_fig(res, model_labels, im):
    """Makes a matplotlib figure with image and classification results.
        Args:
            res: np.array with shape (1, number of labels) containing the results of the classification
                performed by a CNN model.
            im: String path to a single image (the image that generated res)
            model_labels: A python list with the real name of the classes
        Returns:
            fig: a matplotlib figure.
    """
    # for plotting:
    x_pos = [i for i, _ in enumerate(model_labels)]

    # read image for plotting:
    img = mpimg.imread(im)

    # plot image and probabilities:
    fig = plt.figure()
    # set up subplot grid
    gs1 = gridspec.GridSpec(4, 1)

    # plot the image:
    plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=2)
    plt.axis("off")
    plt.imshow(img)

    plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=2)
    plt.barh(x_pos, res[0][:], color='grey')
    plt.ylabel("Class")

    plt.xlabel("Probability")
    # plt.title("Probability assigned by class")
    plt.xlim(0.0, 1.0)
    plt.yticks(x_pos, model_labels)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.set_size_inches(w=14, h=6)

    return fig

"""
Main Graphic class:
"""

class CNN_GUI(tk.Tk):
    """The main class for the graphic interface. Inherits from tkinter.
    """

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="")
        tk.Tk.wm_title(self, "Convolutional Neural Networks")

        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Save settings', command=lambda: popupmsg('Not supported just yet'))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=lambda: quit)
        menubar.add_cascade(label='File', menu=filemenu)

        quick_train = tk.Menu(menubar, tearoff=1)
        quick_train.add_command(label='Inception V3', command=lambda: popupmsg('Not implemented just yet'))
        quick_train.add_command(label='ResNet V2', command=lambda: popupmsg('Not implemented just yet'))
        quick_train.add_command(label='MobileNet V2', command=lambda: popupmsg('Not implemented just yet'))
        quick_train.add_command(label='Quick Train Info', command=lambda: popupmsg('Not implemented just yet'))
        menubar.add_cascade(label='Quick Train', menu=quick_train)

        help_menu = tk.Menu(menubar, tearoff=1)
        help_menu.add_command(label='Info', command=lambda: popupmsg('Not implemented just yet'))
        help_menu.add_command(label='Help', command=lambda: popupmsg('Not implemented just yet'))
        menubar.add_cascade(label='Help', menu=help_menu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, PageSplit, PageTrain, PageTest):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Convolutional Neural Networks\n"
                                    "Transfer Learning User Interface", font=LARGE_FONT)
        label.pack(pady=10, padx=10, fill=tk.BOTH)

        button1 = ttk.Button(self, text="Split Data", width=18, command=lambda: controller.show_frame(PageSplit))
        button1.pack()

        button2 = ttk.Button(self, text="Transfer Learning", width=18, command=lambda: controller.show_frame(PageTrain))
        button2.pack()

        button3 = ttk.Button(self, text="Test Model", width=18, command=lambda: controller.show_frame(PageTest))
        button3.pack()

        button4 = ttk.Button(self, text="Close", width=18, command=lambda: quit)
        button4.pack()

class PageSplit(tk.Frame):
    global validation_per
    global test_per
    global input_folder

    def __init__(self, parent, controller):
        var_validation_per = tk.IntVar()
        var_validation_per.set(10)
        var_test_per = tk.IntVar()
        var_test_per.set(10)

        # aesthetic parameter
        but_width = 20

        tk.Frame.__init__(self, parent)


        label1 = tk.Label(self, text="Split data", font=LARGE_FONT)
        label1.pack(pady=10, padx=10, side=tk.TOP)

        button1 = tk.Button(self, text="Select data folder", width=but_width,
                             command=lambda: folder_select())
        button1.pack(side=tk.TOP)

        label2 = tk.Label(self, text="Select validation percentage: ")
        label2.pack(side=tk.TOP, pady=2, padx=10)

        option1 = tk.OptionMenu(self, var_validation_per, 1, 5, 10, 20, 30)
        option1.pack(side=tk.TOP, pady=2, padx=10)

        label3 = tk.Label(self, text="Select test percentage: ")
        label3.pack(side=tk.TOP, pady=2, padx=10)

        option3 = tk.OptionMenu(self, var_test_per, 0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
        option3.pack(side=tk.TOP, pady=2, padx=10)

        button4 = tk.Button(self, text="Split data", font=NORMAL_FONT_BOLD, width=but_width, command=lambda: split_run())
        button4.pack(side=tk.TOP, pady=10)

        button_home = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button_home.pack(side=tk.BOTTOM)

        def folder_select():
            global input_folder
            input_folder = tk.filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()),
                                                    title='Select data folder')
            print('\nData folder: ' + input_folder + '\n')

        def split_run():
            validation_per = var_validation_per.get()
            test_per = var_test_per.get()

            tl.train_val_test(str(input_folder), validation_per / 100., test_per / 100.)

            popupmsg('Data in ' + str(input_folder) + ' sucessfully split:' +
                     '\n' + str(validation_per) + '% as validation data' +
                     '\n' + str(test_per) + '% as train data')

class PageTrain(tk.Frame):
    global train_folder
    global valid_folder
    def __init__(self, parent, controller):
        var_epochs = tk.IntVar()
        var_epochs.set(10)

        ms = tk.StringVar()                 # model/architecture to be retrained
        var_opt = tk.StringVar()            # optimizer to be used

        # aesthetic parameter
        but_width = 25

        tk.Frame.__init__(self, parent)


        label1 = tk.Label(self, text="Transfer Learning", font=LARGE_FONT)
        label1.pack(pady=10, padx=10, side=tk.TOP, )

        button1 = tk.Button(self, text="Select training data folder", width=but_width,
                             command=lambda: folder_train_select())
        button1.pack(side=tk.TOP)

        button2 = tk.Button(self, text="Select validation data folder", width=but_width,
                            command=lambda: folder_valid_select())
        button2.pack(side=tk.TOP)

        label3 = tk.Label(self, text="Choose one CNN architecture:")
        label3.pack(side=tk.TOP, pady=2, padx=10)

        # for radio button:
        options = [
            'Xception',
            'VGG16',
            'VGG19',
            'ResNet50',
            'InceptionV3',
            'InceptionResNetV2',
            'MobileNet',
            'MobileNetV2',
            'DenseNet121',
            'NASNetLarge'
        ]

        # for model selection parameters
        options_dict = {
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

        ms.set('Xception')  # initialize

        for text in options:
            self.b = tk.Radiobutton(self, text=text, variable=ms, width=but_width, indicatoron=0, borderwidth=3,
                                    value=text, command=lambda: update_arch()).pack()


        label4 = tk.Label(self, text='Enter bottlenecks\' name:')
        label4.pack(side=tk.TOP, pady=2, padx=10)
        entry4 = tk.Entry(self, width=30)
        entry4.insert(tk.END, 'bn')
        entry4.pack(side=tk.TOP, pady=2, padx=10)

        button7 = tk.Button(self, text="Create bottlenecks", font=NORMAL_FONT_BOLD, width=but_width,
                            command=lambda: create_bottlenecks_run())
        button7.pack(side=tk.TOP, pady=10)

        separator1 = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator1.pack(side=tk.TOP, pady=10, padx=10, fill=tk.BOTH)

        label5 = tk.Label(self, text='Enter name for new model:')
        label5.pack(side=tk.TOP, pady=2, padx=10)
        entry5 = tk.Entry(self)
        entry5.insert(tk.END, 'model_1')
        entry5.pack(side=tk.TOP, pady=2, padx=10)

        label6 = tk.Label(self, text="Select number of epochs:")
        label6.pack(side=tk.TOP, pady=2, padx=10)
        option6 = tk.OptionMenu(self, var_epochs, 1, 3, 5, 10, 20, 50, 100, 500, 1000)
        option6.pack(side=tk.TOP, pady=2, padx=10)

        label7 = tk.Label(self, text="Choose one optmizer:")
        label7.pack(side=tk.TOP, pady=2, padx=10)

        var_opt.set('Xception')  # initialize

        optimizers = [
            ('RMSProp', 'RMSprop'),
            ('Stochastic gradient descent', 'SGD')
        ]

        for text, mode in optimizers:
            self.b2 = tk.Radiobutton(self, text=text, variable=var_opt, width=but_width, indicatoron=0, borderwidth=3,
                                    value=mode, command=lambda: update_opt()).pack()


        button8 = tk.Button(self, text="Run transfer learning", font=NORMAL_FONT_BOLD, width=but_width,
                            command=lambda: transfer_learning_run())
        button8.pack(side=tk.TOP, pady=10, padx=10)

        button_home = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button_home.pack(side=tk.BOTTOM)

        def folder_train_select():
            global train_folder
            train_folder = tk.filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()),
                                                    title='Select training data folder')
            print('\nTraining data folder: ' + train_folder)

        def folder_valid_select():
            global valid_folder
            valid_folder = tk.filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()),
                                                    title='Select validation data folder')
            print('\nValidation data folder: ' + valid_folder)

        def update_arch():
            arch = ms.get()
            print("Selected: " + arch + '. The default input size for this model is ' + str(options_dict[arch]))

        def update_opt():
            opt = var_opt.get()
            print("Optimizer: " + opt)

        def create_bottlenecks_run():
            arch = ms.get()
            bn_name = entry4.get()

            tl.save_bottleneck_features(train_folder, valid_folder, bn_name, options_dict[arch][0], options_dict[arch][1], arch)

            popupmsg('Bottlenecks for data in ' + str(train_folder) + ' and ' + str(train_folder) + ' sucessfully created.' +
                     '\nFiles saved in ' + str(bottleneck_dir) + ' as ' + str(bn_name) + '.')

        def transfer_learning_run():
            arch = ms.get()
            bn_name = entry4.get()
            model_name = entry5.get()
            epochs = var_epochs.get()
            opt = var_opt.get()

            tl.train_top_model(bn_name, model_name, arch, options_dict[arch][0], options_dict[arch][1], epochs, opt)

            popupmsg('Transfer learning complete.' +
                     '\nModel  saved in ' + str(model_dir) + ' as ' + str(model_name) + '.')


class PageTest(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Label data with new model", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Select model", width=33,
                            command=lambda: model_select())
        button1.pack(side=tk.TOP, pady=10)

        separator1 = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator1.pack(side=tk.TOP, pady=10, padx=10, fill=tk.BOTH)

        button2 = tk.Button(self, text="Select image to be classified", width=33,
                            command=lambda: image_select())
        button2.pack(side=tk.TOP, pady=10, padx=10)

        separator2 = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator2.pack(side=tk.TOP, pady=10, padx=10, fill=tk.BOTH)

        label3 = tk.Label(self, text='Batch label data with retrained model:')
        label3.pack(side=tk.TOP, pady=(10, 2), padx=10)
        button3 = tk.Button(self, text="Select folder with images to be classified", width=33,
                            command=lambda: folder_select())
        button3.pack(side=tk.TOP, pady=(2, 10), padx=10)

        button4 = tk.Button(self, text="Batch label with new model", font=NORMAL_FONT_BOLD, width=30,
                            command=lambda: do_batch_relabel())
        button4.pack(side=tk.TOP, pady=10, padx=10)

        button_home = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button_home.pack(side=tk.BOTTOM)

        def model_select():
            global new_model
            global new_model_labels
            new_model = tk.filedialog.askopenfilename(initialdir=model_dir,
                                                      title='Select model to be used ', filetypes=[("Model files", "*.hdf5")])
            print('\nModel selected: ' + new_model)

            # associated label dictionary:
            new_model_labels = new_model.split('.')[0] + '_dict_l'

        def image_select():
            global single_im
            global new_model
            global new_model_labels

            single_im = tk.filedialog.askopenfilename(initialdir=os.path.dirname(os.getcwd()),
                                                      title='Select image to be classified ')
            print('\nImage selected: ' + single_im)

            # load the label dictionary:
            with open(new_model_labels, 'rb') as f:
                model_labels = pickle.load(f)

            # print the results:
            res = tl.label_one(single_im, new_model)

            for i in range(len(model_labels)):
                print(model_labels[i], res[0][i])

            popupgraph(make_fig(res, model_labels, single_im))

        def folder_select():
            global folder_im
            folder_im = tk.filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()),
                                                      title='Select model to be used ')
            print('\nFolder to be classified: ' + folder_im)

        def do_batch_relabel():
            global folder_im
            global new_model
            global new_model_labels

            # load the label dictionary:
            with open(new_model_labels, 'rb') as fi:
                model_labels = pickle.load(fi)

            res = tl.label_folder(folder_im, new_model)

            # save results as dataframe
            df = pd.DataFrame(res[0], columns=model_labels)
            df['file'] = res[1]

            # save results to disk
            df.to_csv(os.path.dirname(folder_im) + os.sep + os.path.basename(folder_im) + '.csv')

            print('Results for images in folder \n' + folder_im + '\nsaved in \n' +
                     os.path.dirname(folder_im) + os.sep + os.path.basename(folder_im) + '.csv')

            popupmultigraph(df, folder_im, model_labels)

"""
Execution of graphical interface:
"""
app = CNN_GUI()
app.geometry('720x800')
app.mainloop()
