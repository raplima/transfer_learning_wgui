# Rafael Pires de Lima
# January 2019
# wxWidgets based GUI for _processing.py script

# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jan  9 2019)
## http://www.wxformbuilder.org/
##
###########################################################################

import wx
import wx.xrc
import os

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure


# path variables:
path_input = os.getcwd()

bottleneck_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'bnecks' + os.sep
model_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'models' + os.sep


###########################################################################
## Class MainFrame
###########################################################################

class MainFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="Transfer learning with Convolutional Neural Networks",
                          pos=wx.DefaultPosition,
                          size=wx.Size(540, 600),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        self.menubar = wx.MenuBar()

        self.menu_a = wx.Menu()
        self.menu_a.Append(wx.ID_ABOUT, "About")
        self.menu_a.Append(wx.ID_ABOUT, "Help")

        self.menubar.Append(self.menu_a, "About")

        self.SetMenuBar(menuBar=self.menubar)

        bSizer = wx.BoxSizer(wx.VERTICAL)

        bSizer = wx.BoxSizer(wx.VERTICAL)

        self.m_notebook = wx.Notebook(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_panel1 = wx.Panel(self.m_notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        gbSizer = wx.GridBagSizer(0, 0)
        gbSizer.SetFlexibleDirection(wx.BOTH)
        gbSizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.m_staticText_input = wx.StaticText(self.m_panel1, wx.ID_ANY, u"Select data folder:",
                                                wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_input.Wrap(-1)

        gbSizer.Add(self.m_staticText_input, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_dirPicker_in = wx.DirPickerCtrl(self.m_panel1, wx.ID_ANY, wx.EmptyString, u"Select a folder",
                                               wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)
        self.m_dirPicker_in.SetToolTip(
            u"Select a folder containing subfolders (classes) and each one of the subfolders contains images. ")
        self.m_dirPicker_in.SetInitialDirectory(dir=path_input)

        gbSizer.Add(self.m_dirPicker_in, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticText_valper = wx.StaticText(self.m_panel1, wx.ID_ANY, u"Select validation percentage:",
                                                 wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_valper.Wrap(-1)

        gbSizer.Add(self.m_staticText_valper, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        m_comboBox_vper = [u"1", u"5", u"10", u"20", u"30"]
        self.m_comboBox_vper = wx.ComboBox(self.m_panel1, wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox_vper, 0)
        self.m_comboBox_vper.SetToolTip(u"The percentage of data to be reserved as validation data.")
        gbSizer.Add(self.m_comboBox_vper, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_staticText_testper = wx.StaticText(self.m_panel1, wx.ID_ANY, u"Select test percentage:",
                                                  wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_testper.Wrap(-1)

        gbSizer.Add(self.m_staticText_testper, wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        m_comboBox_tper = [u"0", u"1", u"5", u"10", u"20", u"30", u"40", u"50", u"60", u"70", u"80", u"90", u"95"]
        self.m_comboBox_tper = wx.ComboBox(self.m_panel1, wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize,
                                        m_comboBox_tper, 0)
        self.m_comboBox_tper.SetToolTip(u"The percentage of data to be reserved as test data.")
        self.m_comboBox_tper.SetSelection(3)
        gbSizer.Add(self.m_comboBox_tper, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_staticText_split = wx.StaticText(self.m_panel1, wx.ID_ANY, u"Split data:", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.m_staticText_split.Wrap(-1)

        gbSizer.Add(self.m_staticText_split, wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_button_runSD = wx.Button(self.m_panel1, wx.ID_ANY, u"Run", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button_runSD.SetToolTip(
            u"Press this button after selecting the options above. \nThe program will then use split the data into training, validation, and test sets. ")

        gbSizer.Add(self.m_button_runSD, wx.GBPosition(3, 1), wx.GBSpan(1, 1),
                    wx.ALL | wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND, 5)

        self.m_panel1.SetSizer(gbSizer)
        self.m_panel1.Layout()
        gbSizer.Fit(self.m_panel1)
        self.m_notebook.AddPage(self.m_panel1, u"Split Data", True)
        self.m_panel2 = wx.Panel(self.m_notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                 wx.BORDER_RAISED | wx.TAB_TRAVERSAL)
        self.m_panel2.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
        self.m_panel2.SetToolTip(u"The optimizer to be used.")

        gbSizer1 = wx.GridBagSizer(0, 0)
        gbSizer1.SetFlexibleDirection(wx.BOTH)
        gbSizer1.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.m_staticText_trFolder = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Select training data folder:",
                                                   wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_trFolder.Wrap(-1)

        self.m_staticText_trFolder.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_trFolder, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_dirPicker_trFolder = wx.DirPickerCtrl(self.m_panel2, wx.ID_ANY, wx.EmptyString, u"Select a folder",
                                                     wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)
        self.m_dirPicker_trFolder.SetInitialDirectory(dir=path_input)
        self.m_dirPicker_trFolder.SetToolTip(
            u"Select a folder with subfolders containing images from the same class to be used in training. ")

        gbSizer1.Add(self.m_dirPicker_trFolder, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticText_valFolder = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Select validation data folder:",
                                                    wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_valFolder.Wrap(-1)

        self.m_staticText_valFolder.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_valFolder, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_dirPicker_valFolder = wx.DirPickerCtrl(self.m_panel2, wx.ID_ANY, wx.EmptyString,
                                                      u"Select a folder", wx.DefaultPosition,
                                                      wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)
        self.m_dirPicker_valFolder.SetInitialDirectory(dir=path_input)
        self.m_dirPicker_valFolder.SetToolTip(
            u"Select a folder with subfolders containing images from the same class to be used in validation (validation helps in the analysis of hyperparameters and overfitting). ")

        gbSizer1.Add(self.m_dirPicker_valFolder, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticText_cnnModel = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Choose one CNN model",
                                                   wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_cnnModel.Wrap(-1)

        self.m_staticText_cnnModel.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_cnnModel, wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        m_choice_ModelChoices = [u"Xception", u"VGG16", u"VGG19", u"ResNet50", u"NASNetLarge", u"DenseNet121",
                                 u"MobileNet", u"MobileNetV2", u"InceptionResNetV2", u"InceptionV3"]
        self.m_choice_Model = wx.Choice(self.m_panel2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                        m_choice_ModelChoices, 0)
        self.m_choice_Model.SetSelection(0)
        self.m_choice_Model.SetToolTip(
            u"Choose one of the available CNN models previously trained in the ImageNet data. The model selected here is the base of the model to be generated when running transfer learning. ")

        gbSizer1.Add(self.m_choice_Model, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_staticText_bName = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Enter bottlenecks' name:",
                                                wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_bName.Wrap(-1)

        self.m_staticText_bName.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_bName, wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_textCtrl_bName = wx.TextCtrl(self.m_panel2, wx.ID_ANY, u"bn1", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_textCtrl_bName.SetToolTip(
            u"A tag to be part of the bottlenecks name (for training and validation). This tag is then reused below when runing transfer learning. ")

        gbSizer1.Add(self.m_textCtrl_bName, wx.GBPosition(3, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_staticText_bneck = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Create bottlenecks:", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.m_staticText_bneck.Wrap(-1)

        self.m_staticText_bneck.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_bneck, wx.GBPosition(4, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_button_runBot = wx.Button(self.m_panel2, wx.ID_ANY, u"Run", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button_runBot.SetToolTip(
            u"Press this button after selecting the options above. \nThe program will then save bottlenecks based on the options selected. ")

        gbSizer1.Add(self.m_button_runBot, wx.GBPosition(4, 1), wx.GBSpan(1, 2), wx.ALL | wx.EXPAND, 5)

        self.m_staticline1 = wx.StaticLine(self.m_panel2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                           wx.LI_HORIZONTAL)
        gbSizer1.Add(self.m_staticline1, wx.GBPosition(5, 0), wx.GBSpan(1, 2), wx.EXPAND | wx.ALL, 5)

        self.m_staticText_mName = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Model name", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.m_staticText_mName.Wrap(-1)

        self.m_staticText_mName.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_mName, wx.GBPosition(6, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_textCtrl_mName = wx.TextCtrl(self.m_panel2, wx.ID_ANY, u"bn1_m1", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_textCtrl_mName.SetToolTip(u"The name of the model to be generated. No need to provide extensions. ")

        gbSizer1.Add(self.m_textCtrl_mName, wx.GBPosition(6, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_staticText_nEpochs = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Select number of epochs:",
                                                  wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_nEpochs.Wrap(-1)

        self.m_staticText_nEpochs.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_nEpochs, wx.GBPosition(7, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        m_comboBox_nEpochsChoices = [u"1", u"3", u"5", u"10", u"20", u"50", u"100"]
        self.m_comboBox_nEpochs = wx.ComboBox(self.m_panel2, wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize,
                                              m_comboBox_nEpochsChoices, 0)
        self.m_comboBox_nEpochs.SetSelection(3)
        self.m_comboBox_nEpochs.SetToolTip(u"The number of complete iterations to be performed.")
        gbSizer1.Add(self.m_comboBox_nEpochs, wx.GBPosition(7, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_staticText_wOpt = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Choose one optimizer:", wx.DefaultPosition,
                                               wx.DefaultSize, 0)
        self.m_staticText_wOpt.Wrap(-1)

        self.m_staticText_wOpt.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_wOpt, wx.GBPosition(8, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        m_choice_OptChoices = [u"RMSProp", u"Stochastic gradient descent"]
        self.m_choice_Opt = wx.Choice(self.m_panel2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice_OptChoices,
                                      0)
        self.m_choice_Opt.SetSelection(1)
        self.m_choice_Opt.SetToolTip(
            u"Choose one of the available CNN models previously trained in the ImageNet data. The model selected here is the base of the model to be generated when running transfer learning. ")

        gbSizer1.Add(self.m_choice_Opt, wx.GBPosition(8, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_staticText_wOpt1 = wx.StaticText(self.m_panel2, wx.ID_ANY, u"Run transfer learning:", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.m_staticText_wOpt1.Wrap(-1)

        self.m_staticText_wOpt1.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        gbSizer1.Add(self.m_staticText_wOpt1, wx.GBPosition(9, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_button_runTL = wx.Button(self.m_panel2, wx.ID_ANY, u"Run", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button_runTL.SetToolTip(
            u"Press this button after selecting the options above. \nThe program will then use the bottlenecks indicated to perform transfer learning. ")

        gbSizer1.Add(self.m_button_runTL, wx.GBPosition(9, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_panel2.SetSizer(gbSizer1)
        self.m_panel2.Layout()
        gbSizer1.Fit(self.m_panel2)
        self.m_notebook.AddPage(self.m_panel2, u"Transfer Learning", False)
        self.m_panel3 = wx.Panel(self.m_notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        gbSizer2 = wx.GridBagSizer(0, 0)
        gbSizer2.SetFlexibleDirection(wx.BOTH)
        gbSizer2.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.m_staticText_label = wx.StaticText(self.m_panel3, wx.ID_ANY, u"Label data with new model",
                                                wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL)
        self.m_staticText_label.Wrap(-1)

        gbSizer2.Add(self.m_staticText_label, wx.GBPosition(0, 0),
                     wx.GBSpan(1, 2), wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_staticText_mSelect = wx.StaticText(self.m_panel3, wx.ID_ANY,
                                                  u"Select model:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_mSelect.Wrap(-1)

        gbSizer2.Add(self.m_staticText_mSelect, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_filePicker_model = wx.FilePickerCtrl(self.m_panel3, wx.ID_ANY, wx.EmptyString,
                                                    u"Select the retrained model", u"*.hdf5*",
                                                    wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE)
        self.m_filePicker_model.SetInitialDirectory(dir=model_dir)
        self.m_filePicker_model.SetToolTip(u"Select a retrained CNN model.")

        gbSizer2.Add(self.m_filePicker_model, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticline2 = wx.StaticLine(self.m_panel3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                           wx.LI_HORIZONTAL)
        gbSizer2.Add(self.m_staticline2, wx.GBPosition(2, 0), wx.GBSpan(1, 2), wx.EXPAND | wx.ALL, 5)

        self.m_staticText_iSelect = wx.StaticText(self.m_panel3, wx.ID_ANY, u"Select image to be classified:",
                                                  wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_iSelect.Wrap(-1)

        gbSizer2.Add(self.m_staticText_iSelect, wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_filePicker_image = wx.FilePickerCtrl(self.m_panel3, wx.ID_ANY, wx.EmptyString,
                                                    u"Select an image file to be classified", u"*.*",
                                                    wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE)
        self.m_filePicker_image.SetInitialDirectory(dir=path_input)
        self.m_filePicker_image.SetToolTip(
            u"Select one image that will be classfied with the retrained model selected above. The classification happens automatically after the image is selected.")

        gbSizer2.Add(self.m_filePicker_image, wx.GBPosition(3, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticline21 = wx.StaticLine(self.m_panel3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                            wx.LI_HORIZONTAL)
        gbSizer2.Add(self.m_staticline21, wx.GBPosition(4, 0), wx.GBSpan(1, 2), wx.EXPAND | wx.ALL, 5)

        self.m_staticText_iSelectF = wx.StaticText(self.m_panel3, wx.ID_ANY, u"Select folder to be classified:",
                                                   wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_iSelectF.Wrap(-1)

        gbSizer2.Add(self.m_staticText_iSelectF, wx.GBPosition(5, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_dirPicker_fTest = wx.DirPickerCtrl(self.m_panel3, wx.ID_ANY, wx.EmptyString, u"Select a folder",
                                                  wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)
        self.m_dirPicker_fTest.SetInitialDirectory(dir=path_input)
        self.m_dirPicker_fTest.SetToolTip(
            u"Select a folder containing sublfolders with images to be classified with the retrained model. ")

        gbSizer2.Add(self.m_dirPicker_fTest, wx.GBPosition(5, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_staticText_blabel = wx.StaticText(self.m_panel3, wx.ID_ANY, u"Classify the selected folder:",
                                                 wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText_blabel.Wrap(-1)

        gbSizer2.Add(self.m_staticText_blabel, wx.GBPosition(6, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.m_button_runBLabel = wx.Button(self.m_panel3, wx.ID_ANY, u"Run", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button_runBLabel.SetToolTip(
            u"Press this button after selecting the model and folder to be classified above. \nThe program will then use the model indicated to classify the images in the subfolders. ")

        gbSizer2.Add(self.m_button_runBLabel, wx.GBPosition(6, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        self.m_panel3.SetSizer(gbSizer2)
        self.m_panel3.Layout()
        gbSizer2.Fit(self.m_panel3)
        self.m_notebook.AddPage(self.m_panel3, u"Test Model", False)

        bSizer.Add(self.m_notebook, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(bSizer)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.m_button_runSD.Bind(wx.EVT_BUTTON, self.split_data)
        self.m_button_runBot.Bind(wx.EVT_BUTTON, self.create_bneck)
        self.m_button_runTL.Bind(wx.EVT_BUTTON, self.run_transfer_learning)
        self.m_filePicker_image.Bind(wx.EVT_FILEPICKER_CHANGED, self.label_one)
        self.m_button_runBLabel.Bind(wx.EVT_BUTTON, self.label_folder)

    def __del__(self):
        pass

    # functions
    def split_data(self, event): # (folder,val,test)
        event.Skip()

    def create_bneck(self, event):  # (train_f,val_f,cnn_basel)
        event.Skip()

    def run_transfer_learning(self, event):  # (bneck,cnn_basel, n_epochs)
        event.Skip()

    def label_one(self, event):  # (model, image)
        event.Skip()

    def label_folder(self, event):  # (model, folder)
        event.Skip()

if __name__ == '__main__':
    app = wx.App()
    window = MainFrame(None)
    window.Show(True)
    app.MainLoop()

    print('Main window done')

    wx.MessageBox(message='Message',
                  caption='Caption',
                  style=wx.OK | wx.ICON_INFORMATION)

    print('Dialog (message) window done')

    # Creates just a figure and only one subplot

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1,2,3,4,5], [1,2,3,4,5])
    ax.set_title('Simple test image')
    plt.show()

    print('Figure window done')

