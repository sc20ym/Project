# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QFileDialog, QVBoxLayout, QLabel, QTextEdit, QMenuBar, QComboBox, QTextBrowser
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from logzero import logger
from sklearn.metrics import precision_score, recall_score, accuracy_score
import random
import pandas as pd
import shap
import lime
import matplotlib
matplotlib.use('agg')  # 使用 'agg' 后端
import matplotlib.pyplot as plt
import io

import joblib
import random
random.seed(42)  # 使用任意整数作为种子


class Ui_MainWindow(object):
    """
    改slice_number

    Args:
        object (_type_): _description_
    """
    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(900, 650)
        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 120, 430, 430))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 120, 430, 430))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        
        self.text_fixed = QTextBrowser(self.centralwidget)
        self.text_fixed.setGeometry(QtCore.QRect(20, 70, 100, 30))
        self.text_fixed.setObjectName("select_model")
        #  创建下拉框
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(130, 70, 120, 30))
        self.comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.comboBox.setObjectName("combo")
        self.comboBox.addItems(['lr', 'Random Forest', 'SVM'])  # 添加选项
        self.comboBox.currentIndexChanged.connect(self.select_model)

        self.upload_button = QPushButton(self.centralwidget)
        self.upload_button.setGeometry(QtCore.QRect(750, 60, 130, 50))
        self.upload_button.setObjectName("pushButton")
        self.upload_button.clicked.connect(self.upload_csv)

        # shap预测
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(150, 570, 130, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.draw_shap)
        
        # lime预测
        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(590, 570, 130, 50))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.draw_lime)

        # 选择第几个样本
        self.line_number = QComboBox(self.centralwidget)
        self.line_number.setGeometry(QtCore.QRect(730, 570, 60, 50))
        self.line_number.setObjectName("line_number")
        self.line_number.currentIndexChanged.connect(self.select_sample)
        
        #　类别展示
        self.target_type = QTextBrowser(self.centralwidget)
        self.target_type.setGeometry(QtCore.QRect(800, 570, 80, 50))
        self.target_type.setObjectName("target_type")

        # 中间 评估框
        self.eva_text = QTextBrowser(self.centralwidget)
        self.eva_text.setGeometry(QtCore.QRect(280, 70, 400, 30))
        self.eva_text.setObjectName("eva_text")

        ### 标题
        self.title = QTextBrowser(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(320, 20, 260, 40)) # 左 上 宽 高
        self.title.setStyleSheet("background-color: rgb(222, 222, 222);")
        self.title.setObjectName("title")

        self.MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self.MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 889, 26))
        self.menubar.setObjectName("menubar")
        self.MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.MainWindow.setStatusBar(self.statusbar)

        # 定义一些文件变量
        self.file_name = None
        self.data = None
        self.used_model = 'lr'
        self.outs = None
        self.fea_cols = None
        
        # 定义模型
        self.lr_model = joblib.load('./models/lr.pkl')
        self.rf_model = joblib.load('./models/rf.pkl')
        self.svm_model = joblib.load('./models/svm.pkl')
        self.scaler = joblib.load('./models/scaler.pkl')
        logger.info('加载模型完成')

        self.retranslateUi(self.MainWindow)
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def upload_csv(self):
        """
        上传文件
        """
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self.MainWindow,
                                                  "上传csv文件","","csv Files (*.csv);;All Files (*)",
                                                  options=options)
        self.file_name = fileName
        if fileName:
            logger.info(f"用户选择的文件: {fileName}")
            # 加载csv文件
            self.data = pd.read_csv(fileName, index_col=0).reset_index(drop=True)
            pred_data = self.data.dropna()
            pred_data['Sex'] = pred_data['Sex'].replace({'m': 0, 'f': 1})
            ### 以下这部分不确定要不要
            pred_data['ALB'].fillna(pred_data['ALB'].mean(), inplace=True)
            pred_data['ALP'].fillna(pred_data['ALP'].mean(), inplace=True)
            pred_data['CHOL'].fillna(pred_data['CHOL'].mean(), inplace=True)
            pred_data['PROT'].fillna(pred_data['PROT'].mean(), inplace=True)
            pred_data['ALT'].fillna(pred_data['ALT'].mean(), inplace=True)
            ### 以上
            cols_to_scale = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
            self.fea_cols = [x for x in pred_data.columns if x != 'Category']
            pred_data[cols_to_scale] = self.scaler.transform(pred_data[cols_to_scale])
            self.pred_data = pred_data.dropna()
            if self.used_model == 'svm':
                outs = self.svm_model.predict(pred_data[self.fea_cols])
            elif self.used_model == 'Random Forest':
                outs = self.rf_model.predict(pred_data[self.fea_cols])
            else:
                outs = self.lr_model.predict(pred_data[self.fea_cols])
            logger.info(outs)
            self.outs = outs
            y_true = pred_data['Category'].values.tolist()
            p = precision_score(y_true, outs)
            r = recall_score(y_true, outs)
            acc = accuracy_score(y_true, outs)
            logger.info(f"precision: {p}")
            logger.info(f"recall: {r}")
            logger.info(f"accuracy: {acc * 100}%")
            self.eva_text.setText(f"support：{len(y_true)} precision: {p} recall: {r}, accuracy:{round(acc*100, 2)} %")
            self.line_number.clear()
            self.line_number.addItems([str(x) for x in list(range(1, len(y_true) + 1))])
            pred_data_shap = self.pred_data[self.fea_cols].copy()
            if self.used_model == 'lr':
                self.shap_explainer = shap.LinearExplainer(self.lr_model, pred_data_shap)
            elif self.used_model == 'Random Forest':
                self.shap_explainer = shap.TreeExplainer(self.rf_model)
            else:
                self.shap_explainer = shap.KernelExplainer(self.svm_model.predict, pred_data_shap)
            train_data = pd.read_csv('./train_data.csv')
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(train_data[self.fea_cols], 
                                                    feature_names=self.fea_cols, 
                                                    class_names=[0, 1], 
                                                    mode="classification", 
                                                    feature_selection="highest_weights", 
                                                    discretize_continuous=False,
                                                    random_state=42)


    def select_model(self, index):
        selected_text = self.comboBox.itemText(index)
        logger.info(f'用户选择了: {selected_text}')
        self.used_model = selected_text

    def select_sample(self, index):
        selected_text = self.line_number.itemText(index)
        logger.info(f'用户选择了: {selected_text}')
        self.sample_index = selected_text

    def draw_shap(self):
        """
        画图
        """
        if self.file_name:
            logger.info(f"当前model{self.used_model}")
            pred_data_shap = self.pred_data[self.fea_cols].copy()
            logger.info(pred_data_shap.shape)

            shap_values = self.shap_explainer.shap_values(pred_data_shap)
            plt.figure()
            shap.summary_plot(shap_values, pred_data_shap, feature_names=pred_data_shap.columns)
            # plt.axis('off')  # 关闭坐标轴
            # 将图像保存到字节流
            byte_stream = io.BytesIO()
            plt.savefig(byte_stream, format='PNG', bbox_inches='tight', pad_inches=0.0)
            byte_stream.seek(0)

            # 清除当前的matplotlib图形，避免重复绘制
            plt.clf()
            # 创建一个QImage对象
            image = QImage.fromData(byte_stream.read())
            logger.info(self.label.size())
            scaled_image = image.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmap = QPixmap.fromImage(scaled_image)

            # 设置QLabel的pixmap
            self.label.setPixmap(pixmap)

            # 调整QLabel的大小以适应图像（如果需要）
            # self.label_2.resize(pixmap.size())

    def draw_lime(self):
        """
        画图
        """
        if self.file_name:
            
            pred_data_lime = self.pred_data[self.fea_cols]
            try:
                target_index = int(self.sample_index) - 1
            except:
                target_index = 0 #  出错则默认第一个样本
            logger.info(f"***{target_index}")

            logger.info(f"当前模型{self.used_model}, lime对象：{target_index}")
            out_keys = {1: "阳性", 0: "negative"}
            out_type = out_keys.get(self.outs[target_index])
            logger.info(f"样本{target_index}预测为{out_type}")
            self.target_type.setText(out_type)
            if self.used_model == 'svm':
                exp = self.lime_explainer.explain_instance(pred_data_lime.iloc[target_index], self.svm_model.predict_proba)
            elif self.used_model == 'Random Forest':
                exp = self.lime_explainer.explain_instance(pred_data_lime.iloc[target_index], self.rf_model.predict_proba)
            else:
                exp = self.lime_explainer.explain_instance(pred_data_lime.iloc[target_index], self.lr_model.predict_proba)
            
            fig = exp.as_pyplot_figure()

            # 将 matplotlib 的图像转换为 QPixmap
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG')
            buffer.seek(0)

            # 清除当前的matplotlib图形，避免重复绘制
            plt.clf()
            # 创建一个QImage对象
            image = QImage.fromData(buffer.read())
            logger.info(self.label_2.size())
            scaled_image = image.scaled(self.label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmap = QPixmap.fromImage(scaled_image)

            # 设置QLabel的pixmap
            self.label_2.setPixmap(pixmap)

            # 调整QLabel的大小以适应图像（如果需要）
            # self.label_2.resize(pixmap.size())


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "SHAP"))
        self.label_2.setText(_translate("MainWindow", "LIME"))
        self.eva_text.setText("<p>support： &nbsp;&nbsp;precision: &nbsp;&nbsp;recall：&nbsp;&nbsp;accuracy：&nbsp;&nbsp;</p>")
        self.upload_button.setText(_translate("MainWindow", "open file"))
        self.pushButton_2.setText(_translate("MainWindow", "SHAP"))
        self.pushButton_3.setText(_translate("MainWindow", "LIME"))
        self.text_fixed.setHtml("<p>Model</p>")
        self.title.setHtml("<h2>Disease identification interface</h2>")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
