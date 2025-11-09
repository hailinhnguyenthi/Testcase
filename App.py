from HousingPriceSimpleModel import *
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = HousePriceApp()
    window.show()
    sys.exit(app.exec())