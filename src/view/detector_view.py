from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal

class DetectorView(QWidget):
    """
    Detector view for real-time sign detection.
    """
    load_video_signal = pyqtSignal(str) # New signal to emit video file path

    def __init__(self, viewmodel):
        super().__init__()
        self._viewmodel = viewmodel
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initializes the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Video feed label
        self.video_label = QLabel("Cargando cámara...")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480) # Ensure it can expand
        main_layout.addWidget(self.video_label, 1) # Stretch factor to fill space

        # Detected sign label
        self.sign_label = QLabel("Seña Detectada: --")
        self.sign_label.setObjectName("sign_label")
        self.sign_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.sign_label)

        main_layout.addStretch() # Push content to top

    def connect_signals(self):
        """Connects UI element signals to ViewModel slots."""
        # Connect ViewModel signals to UI slots
        self._viewmodel.frame_ready.connect(self.update_video_feed)
        self._viewmodel.sign_detected.connect(self.update_sign_label)

    @pyqtSlot(QImage)
    def update_video_feed(self, image):
        """Updates the video feed label with a new frame."""
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @pyqtSlot(str)
    def update_sign_label(self, sign):
        """Updates the detected sign label."""
        self.sign_label.setText(f"Seña Detectada: {sign}")
