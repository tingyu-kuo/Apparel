from PyQt5 import QtWidgets, QtGui, QtCore
from main_window import Ui_MainWindow
import os, sys, time
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QMessageBox
from datetime import datetime as dt
from pathlib import Path
import pandas as pd
import predict
import numpy as np
import torch


class Runthread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(str)

    def __init__(self, in_path, out_path):
        super(Runthread, self).__init__()
        self.in_path = in_path
        self.out_path = out_path

    def run(self):
        # 執行判讀及結果處理
        self.preds = predict.Predict(in_path=self.in_path, out_path=self.out_path)
        init = np.full([len(self.preds.dataset), 4], np.nan)
        self.record = pd.DataFrame(init, columns=['file','type','product','color'])
        index = 0
        for i, (inputs, file_n) in enumerate(self.preds.dataloader):
            p = int(((i+1)*2 / len(self.preds.dataset))*100) # 進度百分比
            p = 100 if p>=100 else p
            inputs = inputs.cuda(self.preds.device) if self.preds.use_cuda else inputs
            # type predict
            type_outputs = self.preds.type_model(inputs)
            self.type_preds = torch.argmax(type_outputs, dim=1)
            # product predict
            product_outputs = self.preds.product_model(inputs)
            self.product_preds = torch.argmax(product_outputs, dim=1)
            # color predict
            color_outputs = self.preds.color_model(inputs)
            self.color_preds = torch.argmax(color_outputs, dim=1)

            for i in range(len(inputs)):
                self.record.loc[index+i, 'file'] = file_n[i]
                self.record.loc[index+i, 'type'] = self.preds.type_classes[self.type_preds[i].item()]
                self.record.loc[index+i, 'product'] = self.preds.product_classes[self.product_preds[i].item()]
                self.record.loc[index+i, 'color'] = self.preds.color_classes[self.color_preds[i].item()]
            index = index + len(inputs)
            self._signal.emit(str(p))
        save_path = self.preds.save / 'record.csv'
        self.record.to_csv(str(save_path), index=False)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.input_filename = str()
        self.output_filename = str()
        # path of Desktop
        self.desk = os.path.join(os.path.expanduser("~"), 'Desktop')

        self.ui_main = Ui_MainWindow()
        self.ui_main.setupUi(self)
        # main window initialize
        self.setWindowTitle(" AI判讀程式 ")
        # set icon
        self.icon = QtGui.QIcon()
        self.icon.addPixmap(QtGui.QPixmap("icon.ico"),QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(self.icon)
        # fixed window
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        # button control
        self.ui_main.pushButton_4.hide()
        self.ui_main.pushButton_5.hide()
        # initial button fn
        self.ui_main.pushButton.clicked.connect(self.input_browse)
        self.ui_main.pushButton_2.clicked.connect(self.output_browse)
        self.ui_main.pushButton_3.clicked.connect(self.isStartClick)
        # button of restart or close
        self.ui_main.pushButton_4.clicked.connect(self.isRestartClick)
        self.ui_main.pushButton_5.clicked.connect(sys.exit)
        self.ui_main.progressBar.hide()

    # brower to choose input and output folder
    def input_browse(self):
        self.input_filename = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾', self.desk)
        self.ui_main.lineEdit.setText(self.input_filename)

    def output_browse(self):
        self.output_filename = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾', self.desk)
        self.ui_main.lineEdit_2.setText(self.output_filename)

    # restart when the continue button is clicked
    def isRestartClick(self):
        self.ui_main.progressBar.hide()
        self.ui_main.pushButton.setEnabled(True)
        self.ui_main.pushButton_2.setEnabled(True)
        self.ui_main.pushButton_3.show()
        self.ui_main.pushButton_4.hide()
        self.ui_main.pushButton_5.hide()
        self.ui_main.pushButton_3.setEnabled(True)
        self.input_filename = str()
        self.output_filename = str()
        self.ui_main.lineEdit.setText(self.input_filename)
        self.ui_main.lineEdit_2.setText(self.output_filename)
        self.ui_main.progressBar.setValue(0)

    # if dir is not exist, then create a dir
    def isDirExist(self, target_path):
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

    # if main process finish, do:
    def finish(self):
        QMessageBox.question(self,'完成','AI判讀完成。\t', QMessageBox.Ok, QMessageBox.Ok)
        self.ui_main.lineEdit_2.setEnabled(False)
        self.ui_main.pushButton_3.setEnabled(True)
        self.ui_main.pushButton_3.hide()
        self.ui_main.pushButton_4.show()
        self.ui_main.pushButton_5.show()
        self.show()

    # start_login & call_backlog控制執行緒回傳
    def start_classify(self):
            self.thread = Runthread(self.input_filename, self.output_filename)
            self.thread._signal.connect(self.call_backlog)  # 執行緒連接回UI的動作
            self.thread.start()

    def call_backlog(self, msg):
        self.ui_main.progressBar.setValue(int(msg))
        if int(msg) >= 100:
            self.finish()
            self.thread.quit()

    # action when start button is clicked
    def isStartClick(self):
        # error messages
        if (self.input_filename == str()) and (self.output_filename == str()):
            QMessageBox.question(self,'錯誤','請選擇輸入及輸出的資料夾。\t', QMessageBox.Retry, QMessageBox.Retry)
            self.show()
        elif (self.input_filename == str()):
            QMessageBox.question(self,'錯誤','請選擇輸入的資料夾。\t', QMessageBox.Retry, QMessageBox.Retry)
            self.show()
        elif (self.output_filename == str()):
            QMessageBox.question(self,'錯誤','請選擇輸出的資料夾。\t', QMessageBox.Retry, QMessageBox.Retry)
            self.show()
        # start successfully
        else:
            # 按鍵及輸入框鎖定
            self.ui_main.progressBar.show()
            self.ui_main.pushButton.setEnabled(False)
            self.ui_main.pushButton_2.setEnabled(False)
            self.ui_main.pushButton_3.setEnabled(False)
            self.ui_main.lineEdit.setEnabled(False)
            self.ui_main.lineEdit_2.setEnabled(False)
            # start...
            try:
                time.sleep(0.1)
                self.start_classify()
            except:
                QMessageBox.question(self,'錯誤','輸入的資料有誤，請確認資料正確性。\t', QMessageBox.Retry, QMessageBox.Retry)
                self.show()
                self.isRestartClick()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
