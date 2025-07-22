from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage
import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import urllib.request
import tempfile
import time # Import time for cooldown logic
from src.ubidots_client import UbidotsClient

# Import fall detection service safely - USAR PADDLEDETECTION REAL
try:
    from src.services.fall_detection_service_paddledetection import FallDetectionService
    FALL_DETECTION_AVAILABLE = True
    print("✅ Importando servicio PaddleDetection completo")
except Exception as e:
    print(f"⚠️ PaddleDetection no disponible, probando servicio simple: {e}")
    try:
        from src.services.fall_detection_service_simple import FallDetectionService
        FALL_DETECTION_AVAILABLE = True
        print("✅ Importando servicio simple como respaldo")
    except Exception as e2:
        print(f"❌ Ningún servicio de detección disponible: {e2}")
        FallDetectionService = None
        FALL_DETECTION_AVAILABLE = False

class VideoProcessorWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    sign_detected = pyqtSignal(str)
    status_message = pyqtSignal(str)
    finished = pyqtSignal()
    send_ubidots_data = pyqtSignal(str, float) # New signal to send data to Ubidots
    fall_detected = pyqtSignal(bool, float) # New signal for fall detection

    def __init__(self, model_path, labels_dict, video_path=None):
        super().__init__()
        self.model_path = model_path
        self.labels_dict = labels_dict
        self.video_path = video_path
        self.is_running = False
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1)
        
        # Inicializar MediaPipe Pose para esqueleto
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize fall detection service safely
        self.fall_detection_service = None
        if FALL_DETECTION_AVAILABLE and FallDetectionService is not None:
            try:
                self.fall_detection_service = FallDetectionService()
                if self.fall_detection_service.is_enabled():
                    self.status_message.emit("Servicio de detección de caídas inicializado")
                else:
                    self.status_message.emit("Advertencia: Detección de caídas no disponible")
            except Exception as e:
                print(f"Error inicializando fall detection service: {e}")
                self.fall_detection_service = None
        else:
            print("Fall detection service not available")

    def run(self):
        self.status_message.emit("Iniciando procesador de video...")
        
        # Load the model (optional - camera works without it)
        model_loaded = False
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)['model']
                self.status_message.emit("Modelo cargado exitosamente.")
                model_loaded = True
            except Exception as e:
                self.status_message.emit(f"Advertencia: Error al cargar el modelo: {e}")
                self.model = None
        else:
            self.status_message.emit("Advertencia: Modelo no encontrado. La cámara funcionará sin detección de señas.")
            self.model = None

        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            self.status_message.emit(f"Cargando video: {self.video_path}")
        else:
            cap = cv2.VideoCapture(0)
            self.status_message.emit("Cargando cámara...")

        if not cap.isOpened():
            self.status_message.emit("Error: No se pudo abrir la fuente de video.")
            self.finished.emit()
            return

        self.is_running = True
        self.status_message.emit("Fuente de video iniciada. Detectando señas...")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if self.video_path: # If it's a video file, stop when it ends
                    self.status_message.emit("Video finalizado.")
                else: # If it's a camera, report error
                    self.status_message.emit("Error: No se pudo leer el frame de la cámara.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # ===== DETECCIÓN INTEGRADA: MANOS + POSE + CAÍDAS =====
            fall_detected = False
            fall_confidence = 0.0
            integrated_processing = False
            
            # Intentar usar el servicio integrado (MediaPipe Hands + PaddleHub Pose + PaddleDetection Falling)
            if self.fall_detection_service and self.fall_detection_service.is_enabled():
                try:
                    # El servicio integrado maneja TODO: manos, pose y caídas
                    fall_detected = self.fall_detection_service.detect_fall(frame)
                    integrated_processing = True
                    
                    if fall_detected:
                        fall_details = self.fall_detection_service.get_detection_details(frame)
                        if fall_details:
                            fall_confidence = fall_details[0].get('confidence', 0.0)
                        
                        self.fall_detected.emit(True, fall_confidence)
                        
                except Exception as e:
                    print(f"Error en servicio integrado: {e}")
                    integrated_processing = False
            
            # Si el servicio integrado no está disponible, usar MediaPipe Pose como respaldo
            if not integrated_processing:
                pose_results = self.pose.process(frame_rgb)
                
                if pose_results.pose_landmarks:
                    # Dibujar esqueleto con MediaPipe (respaldo)
                    self.mp_drawing.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=2  # Cyan = respaldo
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2
                        )
                    )
                    cv2.putText(frame, "Esqueleto MediaPipe (respaldo)", 
                              (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Pose tu cuerpo frente a la camara", 
                              (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ===== DETECCIÓN DE MANOS (Solo si no hay servicio integrado) =====
            predicted_sign = ""
            if not integrated_processing and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks (respaldo)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract landmarks for prediction (only if model is loaded)
                    if self.model is not None:
                        data_aux = []
                        x_ = []
                        y_ = []
                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)

                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x - min(x_))
                            data_aux.append(landmark.y - min(y_))
                        
                        if len(data_aux) == 42: # Ensure correct number of features
                            prediction = self.model.predict([np.asarray(data_aux)])
                            predicted_sign = prediction[0]
                            # Get confidence (if your model provides it, e.g., RandomForestClassifier.predict_proba)
                            # We need to find the probability of the predicted class
                            try:
                                probabilities = self.model.predict_proba([np.asarray(data_aux)])[0]
                                # Find the index of the predicted sign in the model's classes_
                                predicted_class_index = list(self.model.classes_).index(predicted_sign)
                                confidence = probabilities[predicted_class_index]
                            except (AttributeError, ValueError): # Model might not have predict_proba or class not found
                                confidence = 1.0 # Default to 1 if confidence not available

                            self.sign_detected.emit(predicted_sign)
                            self.send_ubidots_data.emit(predicted_sign, confidence) # Emit signal with sign and confidence
                        else:
                            self.sign_detected.emit("Detectando...")
                    else:
                        # No model loaded - just show hand detection
                        self.sign_detected.emit("Mano detectada (sin modelo)")
                        
                cv2.putText(frame, "Mano detectada (respaldo)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            elif not integrated_processing:
                self.sign_detected.emit("No se detecta mano")
                cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Convert frame to QImage for display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self.frame_ready.emit(convert_to_Qt_format)

        cap.release()
        self.status_message.emit("Procesador de video detenido.")
        self.finished.emit()

    def stop(self):
        self.is_running = False

class VideoDownloaderWorker(QObject):
    finished = pyqtSignal(str) # Emits path to downloaded file
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        self.progress.emit(f"Descargando video de: {self.url}")
        try:
            # Create a temporary file to save the video
            # Use a common video extension, e.g., .mp4, for compatibility
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file_path = temp_file.name
            temp_file.close()

            urllib.request.urlretrieve(self.url, temp_file_path)
            self.progress.emit(f"Video descargado a: {temp_file_path}")
            self.finished.emit(temp_file_path)
        except Exception as e:
            self.error.emit(f"Error al descargar el video: {e}")
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path) # Clean up partial download
        finally:
            pass # No need to emit finished here, it's handled by error/finished signals

class DetectorViewModel(QObject):
    """
    ViewModel for the Detector View. Handles real-time sign detection.
    """
    # Signals to update the UI
    frame_ready = pyqtSignal(QImage)
    sign_detected = pyqtSignal(str)
    status_message = pyqtSignal(str)
    detector_status_changed = pyqtSignal(bool)
    fall_detected = pyqtSignal(bool, float)  # Nueva señal para detección de caídas

    def __init__(self, model_path="model.p"):
        super().__init__()
        self.model_path = model_path
        self.labels_dict = {0: 'Peligro', 1: 'Todo_OK', 2: 'Ninguna_Sena'}
        self.video_thread = None
        self.video_worker = None
        self.downloader_thread = None
        self.downloader_worker = None
        self._is_detecting = False
        self.ubidots_client = UbidotsClient() # Instantiate UbidotsClient
        self.last_sent_sign = None
        self.last_sent_time = 0
        self.ubidots_cooldown = 2 # seconds

    def start_detection(self):
        """Starts the real-time detection from camera."""
        self._start_detection_source(video_path=None)

    def start_detection_from_video(self, video_path):
        """Starts the detection from a video file."""
        self._start_detection_source(video_path=video_path)

    def start_detection_from_url(self, url):
        if self._is_detecting:
            self.status_message.emit("Ya hay una detección en curso. Deténgala primero.")
            return

        self.status_message.emit(f"Iniciando descarga de video desde URL: {url}")
        self.downloader_worker = VideoDownloaderWorker(url)
        self.downloader_thread = QThread()

        self.downloader_worker.moveToThread(self.downloader_thread)

        self.downloader_thread.started.connect(self.downloader_worker.run)
        self.downloader_worker.finished.connect(self._on_video_downloaded)
        self.downloader_worker.error.connect(self._on_download_error)
        self.downloader_worker.finished.connect(self.downloader_thread.quit)
        self.downloader_worker.error.connect(self.downloader_thread.quit)
        self.downloader_worker.finished.connect(self.downloader_worker.deleteLater)
        self.downloader_worker.error.connect(self.downloader_worker.deleteLater)
        self.downloader_thread.finished.connect(self.downloader_thread.deleteLater)

        self.downloader_worker.progress.connect(self.status_message)

        self.downloader_thread.start()

    def _on_video_downloaded(self, temp_file_path):
        self.status_message.emit(f"Descarga completada. Iniciando detección desde: {temp_file_path}")
        self._start_detection_source(video_path=temp_file_path)

    def _on_download_error(self, message):
        self.status_message.emit(f"Error de descarga: {message}")
        self.detector_status_changed.emit(False) # Ensure UI reflects non-detecting state

    def _start_detection_source(self, video_path):
        if self._is_detecting:
            return

        self.video_worker = VideoProcessorWorker(self.model_path, self.labels_dict, video_path)
        self.video_thread = QThread()

        self.video_worker.moveToThread(self.video_thread)

        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_worker.finished.connect(self.video_worker.deleteLater)
        self.video_thread.finished.connect(self.video_thread.deleteLater)

        self.video_worker.frame_ready.connect(self.frame_ready)
        self.video_worker.sign_detected.connect(self.sign_detected)
        self.video_worker.status_message.connect(self.status_message)
        self.video_worker.send_ubidots_data.connect(self._send_data_to_ubidots) # Connect new signal
        self.video_worker.fall_detected.connect(self._handle_fall_detection) # Connect fall detection signal

        self.video_thread.start()
        self._is_detecting = True
        self.detector_status_changed.emit(True)
        self.status_message.emit("Iniciando detección...")

    def _send_data_to_ubidots(self, sign, confidence):
        current_time = time.time()
        if sign != self.last_sent_sign or (current_time - self.last_sent_time) > self.ubidots_cooldown:
            self.ubidots_client.publish_event(gesture=sign, confidence=confidence)
            self.last_sent_sign = sign
            self.last_sent_time = current_time

    def _handle_fall_detection(self, fall_detected, confidence):
        """Maneja la detección de caídas"""
        if fall_detected:
            # Emitir señal para actualizar UI
            self.fall_detected.emit(True, confidence)
            
            # Enviar evento crítico a Ubidots
            self.ubidots_client.publish_event(
                fall_event=1, 
                help_signal=1,
                gesture="EMERGENCIA_CAIDA",
                confidence=confidence
            )
            
            # Mensaje de estado
            self.status_message.emit(f"¡CAÍDA DETECTADA! Confianza: {confidence:.2f}")
        else:
            self.fall_detected.emit(False, 0.0)

    def stop_detection(self):
        if not self._is_detecting:
            return

        if self.video_worker:
            self.video_worker.stop()
        
        # Also stop downloader if it's running
        if self.downloader_worker and self.downloader_thread and self.downloader_thread.isRunning():
            self.downloader_thread.quit()
            self.downloader_worker.deleteLater()
            self.downloader_thread.deleteLater()
            self.downloader_worker = None
            self.downloader_thread = None

        self._is_detecting = False
        self.detector_status_changed.emit(False)
        self.status_message.emit("Detección detenida.")
