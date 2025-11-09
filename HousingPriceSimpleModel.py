import os
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from FileUtil import FileUtil  # giữ nguyên module của bạn
from main_window import Ui_MainWindow  # file pyuic6 sinh ra từ giao diện Qt Designer


# -----------------------------
# LỚP CỬA SỔ XEM DATASET
# -----------------------------
class DataViewer(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Xem Dataset")
        self.resize(900, 500)

        # Table widget
        self.table = QtWidgets.QTableWidget()
        self.setCentralWidget(self.table)

        # Thiết lập dữ liệu DataFrame vào bảng
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.table.setItem(i, j, item)

        # Tự điều chỉnh kích thước
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)

        # Style nhẹ cho bảng
        self.table.setStyleSheet("""
            QHeaderView::section {
                background-color: #00bfff;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #dddddd;
            }
            QTableWidget {
                gridline-color: #cccccc;
                font-size: 10pt;
                selection-background-color: #cceeff;
            }
        """)


# -----------------------------
# LỚP CHÍNH CỦA ỨNG DỤNG
# -----------------------------
class HousePriceApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.df = None
        self.lm = None
        self.file_path = ""
        self.setWindowTitle("Nguyen Thi Hai Linh - House Price Prediction (PyQt6)")

        self.btnPick.clicked.connect(self.do_pick_data)
        self.btnView.clicked.connect(self.do_view_dataset)
        self.btnTrain.clicked.connect(self.do_train)
        self.btnEval.clicked.connect(self.do_evaluation)
        self.btnSave.clicked.connect(self.do_save_model)
        self.btnLoad.clicked.connect(self.do_load_model)
        # self.btnPredict.clicked.connect(self.do_prediction)

        # Hiển thị model trong combobox
        self.load_model_files()

    # --- Chọn dataset ---
    def do_pick_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Dataset",
            "",
            "Dataset (*.csv);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            QMessageBox.information(self, "Thông báo", f"Đã chọn file:\n{file_path}")

    # --- Xem dataset ---
    def do_view_dataset(self):
        if not self.file_path:
            QMessageBox.warning(self, "Lỗi", "Bạn cần chọn dataset trước!")
            return
        try:
            df = pd.read_csv(self.file_path)
            df_preview = df.head(100)  # chỉ hiển thị 100 dòng đầu
            self.viewer = DataViewer(df_preview)
            self.viewer.show()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc file: {str(e)}")

    # --- Huấn luyện mô hình ---
    def do_train(self):
        if not self.file_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn dataset trước!")
            return
        try:
            ratio = 0.8  # mặc định 80% huấn luyện
            self.df = pd.read_csv(self.file_path)

            self.X = self.df[['Avg. Area Income', 'Avg. Area House Age',
                              'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                              'Area Population']]
            self.y = self.df['Price']

            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=1 - ratio, random_state=101
            )

            self.lm = LinearRegression()
            self.lm.fit(X_train, y_train)

            self.X_test, self.y_test = X_test, y_test

            QMessageBox.information(self, "Huấn luyện", "Huấn luyện mô hình hoàn tất!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi huấn luyện: {str(e)}")

    # --- Đánh giá mô hình ---
    def do_evaluation(self):
        if self.lm is None:
            QMessageBox.warning(self, "Lỗi", "Bạn cần huấn luyện mô hình trước!")
            return
        try:
            predictions = self.lm.predict(self.X_test)

            coeff_df = pd.DataFrame(
                self.lm.coef_,
                self.X.columns,
                columns=['Coefficient']
            )

            # Hiển thị hệ số
            self.textCoefficients.setPlainText(coeff_df.to_string())

            # Hiển thị chỉ số đánh giá
            mae = metrics.mean_absolute_error(self.y_test, predictions)
            mse = metrics.mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)

            self.labelMAE.setText(f"MAE: {mae:.2f}")
            self.labelMSE.setText(f"MSE: {mse:.2f}")
            self.labelRMSE.setText(f"RMSE: {rmse:.2f}")

            QMessageBox.information(self, "Đánh giá", "Đánh giá mô hình hoàn tất!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi đánh giá: {str(e)}")

    # --- Lưu mô hình ---
    def do_save_model(self):
        if self.lm is None:
            QMessageBox.warning(self, "Lỗi", "Bạn cần huấn luyện mô hình trước khi lưu!")
            return
        try:
            filename = f"housingmodel_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
            file_path = os.path.join("model", filename)
            os.makedirs("model", exist_ok=True)
            FileUtil.savemodel(self.lm, file_path)

            self.comboModels.addItem(filename)
            QMessageBox.information(self, "Lưu mô hình", f"Đã lưu mô hình:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi lưu mô hình: {str(e)}")

    # --- Tải danh sách mô hình ---
    def load_model_files(self):
        self.comboModels.clear()
        self.comboModels.addItem("----Chọn mô hình----")

        if not os.path.exists("model"):
            return
        model_files = [f for f in os.listdir("model") if f.endswith(".zip")]
        for f in model_files:
            self.comboModels.addItem(f)

    # --- Nạp mô hình ---
    def do_load_model(self):
        selected = self.comboModels.currentText()
        if selected == "----Chọn mô hình----":
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn mô hình để nạp!")
            return
        try:
            self.lm = FileUtil.loadmodel(os.path.join("model", selected))
            QMessageBox.information(self, "Nạp mô hình", "Đã nạp mô hình thành công!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi nạp mô hình: {str(e)}")

