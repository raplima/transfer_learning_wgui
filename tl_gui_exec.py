# Rafael Pires de Lima
# January 2019
# Script accesses the _gui_wx.py and processes the data using _processing.py

import wx
import os
import pickle
import pandas as pd

# import the GUI and main processing program
import tl_gui_wx
import tl_processing as tl

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

# path variables:
path_input = os.getcwd()

bottleneck_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'bnecks' + os.sep
model_dir = os.getcwd() + os.sep + 'runs' + os.sep + 'models' + os.sep


###########################################################################
# Main window
###########################################################################
class Exec(tl_gui_wx.MainFrame):
    def __init__(self, parent):
        tl_gui_wx.MainFrame.__init__(self, parent)

    def split_data(self, event):
        print(self.m_dirPicker_in.GetPath())
        print(self.m_comboBox_vper.GetValue())
        print(self.m_comboBox_tper.GetValue())

        # call data split function from tl:
        tl.train_val_test(os.path.abspath(self.m_dirPicker_in.GetPath()),
                          float(self.m_comboBox_vper.GetValue()) / 100.,
                          float(self.m_comboBox_tper.GetValue()) / 100.)

        wx.MessageBox(caption='Process complete',
                      message='Data in {} sucessfully split: \n{}% as validation data, \n{}% as test data.'.format(
                          os.path.abspath(self.m_dirPicker_in.GetPath()),
                          self.m_comboBox_vper.GetValue(),
                          self.m_comboBox_tper.GetValue()),
                      style=wx.OK | wx.ICON_INFORMATION)

    def create_bneck(self, event):
        print(self.m_dirPicker_trFolder.GetPath())
        print(self.m_dirPicker_valFolder.GetPath())
        print(self.m_choice_Model.GetString(self.m_choice_Model.GetCurrentSelection()))
        print(self.m_textCtrl_bName.GetValue())

        # to facilitate future calls, save the model architecture name:
        arch = self.m_choice_Model.GetString(self.m_choice_Model.GetCurrentSelection())

        # call bottleneck creation function from tl:
        print("Initiating feature extraction...")
        tl.save_bottleneck_features(self.m_dirPicker_trFolder.GetPath(),
                                    self.m_dirPicker_valFolder.GetPath(),
                                    self.m_textCtrl_bName.GetValue(),
                                    options_dict[arch][0],
                                    options_dict[arch][1],
                                    arch)
        print("Process complete.")

        wx.MessageBox(caption='Bottlenecks created complete',
                      message='Bottlenecks for data in {} and {} successfully created. \nFiles saved in {}.'.format(
                          self.m_dirPicker_trFolder.GetPath(), self.m_dirPicker_valFolder.GetPath(), bottleneck_dir),
                      style=wx.OK | wx.ICON_INFORMATION)

    def run_transfer_learning(self, event):
        print(self.m_textCtrl_bName.GetValue())
        print(self.m_textCtrl_mName.GetValue())
        print(self.m_comboBox_nEpochs.GetValue())
        print(self.m_choice_Opt.GetString(self.m_choice_Opt.GetCurrentSelection()))

        # to facilitate future calls, save the model architecture name:
        arch = self.m_choice_Model.GetString(self.m_choice_Model.GetCurrentSelection())

        # call bottleneck creation function from tl:
        print("Initiating training...")
        tl.train_top_model(self.m_textCtrl_bName.GetValue(),
                           self.m_textCtrl_mName.GetValue(),
                           arch,
                           options_dict[arch][0],
                           options_dict[arch][1],
                           int(self.m_comboBox_nEpochs.GetValue()),
                           self.m_choice_Opt.GetString(self.m_choice_Opt.GetCurrentSelection()))
        print("Training complete.")

        wx.MessageBox(caption='Training complete',
                      message='Transfer learning complete. \n'
                              'Model saved in {} as {} '.format(model_dir,
                                                                self.m_textCtrl_mName.GetValue()),
                      style=wx.OK | wx.ICON_INFORMATION)

    def label_one(self, event):  # (model, image)
        print(self.m_filePicker_model.GetPath())
        print(self.m_filePicker_image.GetPath())

        # load the label dictionary:
        new_model_labels = self.m_filePicker_model.GetPath().split('.')[0] + '_dict_l'
        with open(new_model_labels, 'rb') as f:
            model_labels = pickle.load(f)

        # get the classification provided by the retrained model:
        res = tl.label_one(self.m_filePicker_image.GetPath(),
                           self.m_filePicker_model.GetPath())

        print('{:50} {:}'.format('Class', 'Probability'))
        for i in range(len(model_labels)):
            print('{:50} {:.4f}'.format(model_labels[i], res[0][i]))

        tl.make_fig(res,
                    model_labels,
                    self.m_filePicker_image.GetPath())

    def label_folder(self, event):  # (model, folder)
        print(self.m_filePicker_model.GetPath())
        print(self.m_dirPicker_fTest.GetPath())

        # save strings to facilitate readability:
        folder_im = self.m_dirPicker_fTest.GetPath()
        new_model = self.m_filePicker_model.GetPath()
        # associated label dictionary:
        new_model_labels = new_model.split('.')[0] + '_dict_l'
        with open(new_model_labels, 'rb') as fi:
            model_labels = pickle.load(fi)

        # call function in tl_processing:
        res = tl.label_folder(folder_im, new_model)

        # save results as dataframe
        df = pd.DataFrame(res[0], columns=model_labels)
        df['file'] = res[1]

        # save results to disk
        _, tail = os.path.split(new_model)
        df.to_csv(folder_im + os.sep + tail.split('.')[0] + '.csv')

        wx.MessageBox(caption='Folder classified',
                      message='Results for images in folder \n'
                              '{} saved in \n {}.csv'.format(folder_im,
                                                             folder_im + os.sep + os.path.basename(folder_im) +
                                                             new_model.split('.')[0]),
                      style=wx.OK | wx.ICON_INFORMATION)


if __name__ == '__main__':
    app = wx.App(False)
    frame = Exec(None)
    frame.Show(True)
    # start the applications
    app.MainLoop()
