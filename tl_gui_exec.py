# Rafael Pires de Lima
# January 2019
# Script accesses the _gui_wx.py and processes the data using _processing.py

import wx

# import the newly created GUI file
import tl_gui_wx

class Exec(tl_gui_wx.MainFrame):
    def __init__(self, parent):
        tl_gui_wx.MainFrame.__init__(self, parent)

    def split_data(self, event):
        print(self.m_dirPicker_in.GetPath())
        print(self.m_comboBox_vper.GetValue())
        print(self.m_comboBox_tper.GetValue())

    def create_bneck(self, event):
        print(self.m_dirPicker_trFolder.GetPath())
        print(self.m_dirPicker_valFolder.GetPath())
        print(self.m_choice_Model.GetString(self.m_choice_Model.GetCurrentSelection()))
        print(self.m_textCtrl_bName.GetValue())

    def run_transfer_learning(self, event):
        print(self.m_textCtrl_bName.GetValue())
        print(self.m_textCtrl_mName.GetValue())
        print(self.m_comboBox_nEpochs.GetValue())
        print(self.m_choice_Opt.GetString(self.m_choice_Opt.GetCurrentSelection()))

    def label_one(self, event):  # (model, image)
        print(self.m_filePicker_model.GetPath())
        print(self.m_filePicker_image.GetPath())

    def label_folder(self, event):  # (model, folder)
        print(self.m_filePicker_model.GetPath())
        print(self.m_dirPicker_fTest.GetPath())

if __name__ == '__main__':
    app = wx.App(False)
    frame = Exec(None)
    frame.Show(True)
    # start the applications
    app.MainLoop()
