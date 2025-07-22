import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QCheckBox, QStatusBar, QFileDialog, QInputDialog
)
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from PyQt6.QtCore import Qt, QSize
import qtawesome as qta
import os

from src.view.admin_view import AdminView
from src.view.detector_view import DetectorView 

from src.viewmodel.admin_viewmodel import AdminViewModel
from src.viewmodel.detector_viewmodel import DetectorViewModel

class MainWindow(QMainWindow):
    """
    Vista principal de la aplicación, reimaginada con un diseño moderno.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Señas - Intecto")
        self.setGeometry(100, 100, 1280, 720)
        self.setMinimumSize(800, 600)

        # --- ViewModels ---
        self.admin_viewmodel = AdminViewModel(model=None) # Model will be passed later if needed
        self.detector_viewmodel = DetectorViewModel()

        # --- Status Bar ---
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Estado: Inactivo")
        self.status_bar.addPermanentWidget(self.status_label)

        # --- Central Widget and Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. Video Panel (Left) ---
        video_panel = QFrame()
        video_panel.setFrameShape(QFrame.Shape.NoFrame)
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(10)
        
        # Integrate DetectorView here
        self.detector_view = DetectorView(self.detector_viewmodel)
        video_layout.addWidget(self.detector_view)

        # --- 2. Control Panel (Right) ---
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel") # For QSS styling
        control_panel.setFrameShape(QFrame.Shape.NoFrame)
        control_panel.setFixedWidth(300)
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        control_title = QLabel("Panel de Control")
        control_title.setObjectName("titleLabel")
        control_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(control_title)

        # Control Buttons for Detector
        self.start_detection_button = self.create_control_button(
            "Iniciar Detección", "fa5s.play-circle", "#28a745", "#FFFFFF"
        )
        self.stop_detection_button = self.create_control_button(
            "Detener Detección", "fa5s.stop-circle", "#dc3545", "#FFFFFF"
        )
        self.stop_detection_button.setEnabled(False)

        control_layout.addWidget(self.start_detection_button)
        control_layout.addWidget(self.stop_detection_button)

        # Load Video Button
        self.load_video_button = self.create_control_button(
            "Cargar Video", "fa5s.video", "#007ACC", "#FFFFFF"
        )
        control_layout.addWidget(self.load_video_button)

        # Load Video from URL Button
        self.load_video_url_button = self.create_control_button(
            "Cargar Video desde URL", "fa5s.link", "#6f42c1", "#FFFFFF"
        )
        control_layout.addWidget(self.load_video_url_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(separator)

        # Admin Button
        self.admin_button = self.create_control_button(
            "Administración", "fa5s.cog", "#6c757d", "#FFFFFF"
        )
        control_layout.addWidget(self.admin_button)

        control_layout.addStretch() # Pushes content to the top

        # --- Final Assembly ---
        main_layout.addWidget(video_panel, 1) # Stretch factor 1
        main_layout.addWidget(control_panel)
        
        # --- Connect Signals ---
        self.start_detection_button.clicked.connect(self.detector_viewmodel.start_detection)
        self.stop_detection_button.clicked.connect(self.detector_viewmodel.stop_detection)
        self.load_video_button.clicked.connect(self.open_video_file_dialog)
        self.load_video_url_button.clicked.connect(self.open_url_input_dialog)
        self.admin_button.clicked.connect(self.open_admin_window)

        # Connect ViewModel signals to UI (MainWindow) slots
        self.detector_viewmodel.detector_status_changed.connect(self.on_detector_status_changed)
        self.detector_viewmodel.status_message.connect(self.update_status_bar)

        # Connect DetectorView signals to DetectorViewModel slots
        self.detector_view.load_video_signal.connect(self.detector_viewmodel.start_detection_from_video)

        # --- Apply Styles ---
        self.setup_styles()
        self.set_status_bar_color("normal") # Initial status bar color

    def create_control_button(self, text, icon_name, bg_color, text_color):
        button = QPushButton(text)
        button.setIcon(qta.icon(icon_name, color=text_color))
        button.setIconSize(QSize(24, 24))
        button.setMinimumHeight(50)
        
        base_color = QColor(bg_color)
        hover_color = base_color.lighter(120).name()
        pressed_color = base_color.darker(120).name()

        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 5px;
                border-radius: 4px;
                font-size: 11pt;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover_color}; }}
            QPushButton:pressed {{ background-color: {pressed_color}; }}
            QPushButton:disabled {{ background-color: #555555; color: #888888; }}
        """)
        return button

    def set_status_bar_color(self, state):
        palette = self.status_bar.palette()
        text_color = QColor("white")

        if state == "normal":
            palette.setColor(QPalette.ColorRole.Window, QColor("#17a2b8"))
            self.status_label.setText("Estado: Inactivo")
        elif state == "detecting":
            palette.setColor(QPalette.ColorRole.Window, QColor("#ffc107"))
            text_color = QColor("black")
            self.status_label.setText("Estado: Detectando...")
        elif state == "error":
            palette.setColor(QPalette.ColorRole.Window, QColor("#dc3545"))
            self.status_label.setText("Estado: Error")
        
        palette.setColor(QPalette.ColorRole.WindowText, text_color)
        self.status_bar.setAutoFillBackground(True)
        self.status_bar.setPalette(palette)
        
    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2B2B2B; }
            QFrame#controlPanel { background-color: #3C3C3C; border-left: 1px solid #454545; }
            QLabel { color: #EAEAEA; font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; }
            QLabel#titleLabel { font-size: 16pt; font-weight: bold; color: #00A1E4; margin-bottom: 10px; }
            QLabel#video_label { background-color: #000; border: 1px solid #454545; border-radius: 4px; }
            QCheckBox { spacing: 8px; color: #EAEAEA; font-size: 10pt; }
            QCheckBox::indicator { width: 18px; height: 18px; }
            QCheckBox::indicator:unchecked { border: 1px solid #777777; background-color: #2B2B2B; border-radius: 3px; }
            QCheckBox::indicator:hover { border: 1px solid #007ACC; }
            QCheckBox::indicator:checked { background-color: #007ACC; border: 1px solid #005A9E; border-radius: 3px; }
            QFrame[frameShape="4"] { border: none; background-color: #555555; height: 1px; }
        """)

    def open_admin_window(self):
        self.admin_window = QMainWindow() # AdminView will be a QMainWindow now
        self.admin_window.setWindowTitle("Administración de Datos y Modelo")
        self.admin_window.setGeometry(200, 200, 800, 600)
        
        admin_central_widget = QWidget()
        admin_layout = QVBoxLayout(admin_central_widget)
        self.admin_view = AdminView(self.admin_viewmodel)
        admin_layout.addWidget(self.admin_view)
        self.admin_window.setCentralWidget(admin_central_widget)

        # Apply the same styles to the admin window
        self.admin_window.setStyleSheet(self.styleSheet())

        self.admin_window.show()
        self.destroyed.connect(self.admin_window.close)

    def open_video_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Video (*.mp4 *.avi *.mov *.mkv)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                video_path = selected_files[0]
                self.detector_viewmodel.start_detection_from_video(video_path)

    def open_url_input_dialog(self):
        url, ok = QInputDialog.getText(self, "Cargar Video desde URL", "Introduce la URL del video:")
        if ok and url:
            self.detector_viewmodel.start_detection_from_url(url)

    def on_detector_status_changed(self, is_detecting):
        self.start_detection_button.setEnabled(not is_detecting)
        self.stop_detection_button.setEnabled(is_detecting)
        if is_detecting:
            self.set_status_bar_color("detecting")
        else:
            self.set_status_bar_color("normal")

    def update_status_bar(self, message):
        self.status_label.setText(f"Estado: {message}")
        if "Error" in message:
            self.set_status_bar_color("error")
        elif "Iniciando" in message or "Detectando" in message:
            self.set_status_bar_color("detecting")
        else:
            self.set_status_bar_color("normal")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())