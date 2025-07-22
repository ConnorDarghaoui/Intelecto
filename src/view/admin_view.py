from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, QFrame
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import pyqtSlot, Qt, QSize
import qtawesome as qta

class AdminView(QWidget):
    """
    Admin view for data capture, dataset creation, and model training.
    Follows a guided workflow.
    """
    def __init__(self, viewmodel):
        super().__init__()
        self._viewmodel = viewmodel
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initializes the user interface."""
        main_layout = QHBoxLayout(self) # Changed to QHBoxLayout for side-by-side layout

        # --- Left Panel: Video Feed ---
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(10)

        # Video feed label
        self.video_label = QLabel("Cámara para Captura")
        self.video_label.setObjectName("video_label") # Use objectName for QSS styling
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480) # Standard webcam resolution
        left_panel_layout.addWidget(self.video_label, 1) # Stretch factor

        main_layout.addLayout(left_panel_layout)

        # --- Right Panel: Dataset, Training, and Log ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(10)

        # --- 1. Capture Controls Group ---
        capture_group = QGroupBox("Captura de Datos")
        capture_layout = QVBoxLayout()

        # Input for number of people
        num_people_layout = QHBoxLayout()
        self.num_people_label = QLabel("Número de personas a registrar:")
        self.num_people_input = QLineEdit("1")
        self.num_people_input.setFixedWidth(50)
        num_people_layout.addWidget(self.num_people_label)
        num_people_layout.addWidget(self.num_people_input)
        num_people_layout.addStretch()
        capture_layout.addLayout(num_people_layout)
        
        # Capture buttons for specific categories
        capture_category_layout = QHBoxLayout()
        self.capture_peligro_button = QPushButton("Capturar Peligro")
        self._style_button(self.capture_peligro_button, "fa5s.exclamation-triangle", "#ffc107", "#000000") # Yellow for warning

        self.capture_todo_ok_button = QPushButton("Capturar Todo OK")
        self._style_button(self.capture_todo_ok_button, "fa5s.check-circle", "#28a745", "#FFFFFF") # Green for OK

        self.capture_none_button = QPushButton("Capturar Ninguna Seña")
        self._style_button(self.capture_none_button, "fa5s.ban", "#6c757d", "#FFFFFF") # Gray for none

        capture_category_layout.addWidget(self.capture_peligro_button)
        capture_category_layout.addWidget(self.capture_todo_ok_button)
        capture_category_layout.addWidget(self.capture_none_button)

        # Cancel button
        self.cancel_capture_button = QPushButton("Cancelar Captura")
        self._style_button(self.cancel_capture_button, "fa5s.stop", "#dc3545", "#FFFFFF")
        self.cancel_capture_button.setEnabled(False)

        capture_layout.addLayout(capture_category_layout)
        capture_layout.addWidget(self.cancel_capture_button)
        capture_group.setLayout(capture_layout)
        right_panel_layout.addWidget(capture_group)

        # --- 2. Dataset Group ---
        dataset_group = QGroupBox("Creación del Dataset")
        dataset_layout = QVBoxLayout()
        self.create_dataset_button = QPushButton("Crear Dataset")
        self._style_button(self.create_dataset_button, "fa5s.database", "#007ACC", "#FFFFFF")
        self.create_dataset_button.setEnabled(False)
        dataset_layout.addWidget(self.create_dataset_button)
        dataset_group.setLayout(dataset_layout)
        right_panel_layout.addWidget(dataset_group)

        # --- 3. Training Group ---
        training_group = QGroupBox("Entrenamiento del Modelo")
        training_layout = QVBoxLayout()
        self.train_model_button = QPushButton("Entrenar Modelo")
        self._style_button(self.train_model_button, "fa5s.brain", "#6f42c1", "#FFFFFF")
        self.train_model_button.setEnabled(False)
        training_layout.addWidget(self.train_model_button)
        training_group.setLayout(training_layout)
        right_panel_layout.addWidget(training_group)

        # --- 4. Log Group ---
        log_group = QGroupBox("Log de Actividad")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        right_panel_layout.addWidget(log_group, 1) # Stretch factor

        main_layout.addLayout(right_panel_layout)

    def _style_button(self, button, icon_name, bg_color, text_color):
        button.setIcon(qta.icon(icon_name, color=text_color))
        button.setIconSize(QSize(24, 24))
        button.setMinimumHeight(45) # Make buttons taller
        
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
                font-size: 10pt;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover_color}; }}
            QPushButton:pressed {{ background-color: {pressed_color}; }}
            QPushButton:disabled {{ background-color: #555555; color: #888888; }}
        """)
        return button

    def connect_signals(self):
        """Connects UI element signals to ViewModel slots."""
        self.capture_peligro_button.clicked.connect(self.start_capture_peligro)
        self.capture_todo_ok_button.clicked.connect(self.start_capture_todo_ok)
        self.capture_none_button.clicked.connect(self.start_capture_none)
        self.cancel_capture_button.clicked.connect(self._viewmodel.cancel_capture)
        self.create_dataset_button.clicked.connect(self._viewmodel.create_dataset)
        self.train_model_button.clicked.connect(self._viewmodel.train_model)

        # Connect ViewModel signals to UI slots
        self._viewmodel.log_message.connect(self.append_log)
        self._viewmodel.enable_create_dataset_button.connect(self.enable_create_dataset)
        self._viewmodel.enable_train_model_button.connect(self.enable_train_model)
        self._viewmodel.capture_started.connect(self.on_capture_started)
        self._viewmodel.frame_ready.connect(self.update_video_feed) # Connect new frame signal

    @pyqtSlot(QImage)
    def update_video_feed(self, image):
        """Updates the video feed label with a new frame."""
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @pyqtSlot(str)
    def append_log(self, message):
        """Appends a message to the log output."""
        self.log_output.append(message)

    @pyqtSlot(bool)
    def enable_create_dataset(self, enabled):
        """Enables or disables the create dataset button."""
        self.create_dataset_button.setEnabled(enabled)
        if enabled:
            self.append_log("===> ACCIÓN REQUERIDA: Presiona 'Crear Dataset' para continuar.")
            self.capture_peligro_button.setEnabled(True)
            self.capture_todo_ok_button.setEnabled(True)
            self.capture_none_button.setEnabled(True)
            self.cancel_capture_button.setEnabled(False)

    @pyqtSlot(bool)
    def enable_train_model(self, enabled):
        """Enables or disables the train model button."""
        self.train_model_button.setEnabled(enabled)
        if enabled:
            self.append_log("===> ACCIÓN REQUERIDA: Presiona 'Entrenar Modelo' para finalizar.")

    def start_capture_peligro(self):
        """Gets number of people and starts capture for 'Peligro' via ViewModel."""
        self._start_capture_with_category("Peligro")

    def start_capture_todo_ok(self):
        """Gets number of people and starts capture for 'Todo_OK' via ViewModel."""
        self._start_capture_with_category("Todo_OK")

    def start_capture_none(self):
        """Gets number of people and starts capture for 'Ninguna_Sena' via ViewModel."""
        self._start_capture_with_category("Ninguna_Sena")

    def _start_capture_with_category(self, category):
        try:
            num_people = int(self.num_people_input.text())
            if num_people > 0:
                self._viewmodel.start_capture(num_people, category)
            else:
                self.append_log("Error: El número de personas debe ser mayor a 0.")
        except ValueError:
            self.append_log("Error: Ingresa un número válido de personas.")

    @pyqtSlot()
    def on_capture_started(self):
        """Disables capture buttons and enables cancel button."""
        self.capture_peligro_button.setEnabled(False)
        self.capture_todo_ok_button.setEnabled(False)
        self.capture_none_button.setEnabled(False)
        self.cancel_capture_button.setEnabled(True)
