import sys
from PyQt6.QtWidgets import QApplication
from src.view.main_window import MainWindow

def main():
    """
    The main entry point of the application.
    Initializes the Qt application and shows the main window.
    """
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
