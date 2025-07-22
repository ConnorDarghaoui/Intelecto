from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage
import os
import cv2
import time
import pickle
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Worker classes to run tasks in separate threads
class CaptureWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    capture_finished = pyqtSignal()
    frame_ready = pyqtSignal(QImage) # New signal for video frames

    def __init__(self, num_people, category):
        super().__init__()
        self.num_people = num_people
        self.category = category
        self.is_running = True
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    def run(self):
        """Captures images for the dataset."""
        self.progress.emit(f"Iniciando captura para {self.num_people} persona(s)...")
        
        dataset_size = 100 # Number of images to capture per category

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.progress.emit("Error: No se pudo abrir la cámara.")
            self.finished.emit()
            return

        self.progress.emit(f"Iniciando captura para la categoría: {self.category}...")
        
        class_path = os.path.join("data", self.category) # Use "data" as base directory
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        self.progress.emit(f'Listo para capturar para la categoría: {self.category}. Preparate...')
        
        # Give a small delay for the user to prepare after the message
        time.sleep(2) 
        
        counter = 0
        while counter < dataset_size:
            if not self.is_running:
                break
            ret, frame = cap.read()
            if not ret:
                self.progress.emit("Error: No se pudo leer el frame de la cámara.")
                break
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Convert frame to QImage for display in UI
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self.frame_ready.emit(convert_to_Qt_format)

            img_path = os.path.join(class_path, f'{int(time.time() * 1000)}.jpg')
            cv2.imwrite(img_path, frame)

            counter += 1
            self.progress.emit(f"Capturando imagen {counter}/{dataset_size} para {self.category}")
            time.sleep(0.1) # Small delay to avoid capturing too fast

        cap.release()
        cv2.destroyAllWindows()
        if self.is_running:
            self.progress.emit("Captura de datos completada.")
            self.capture_finished.emit()
        else:
            self.progress.emit("Captura de datos cancelada.")
        self.finished.emit()

    def stop(self):
        self.is_running = False

class DatasetWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    dataset_created = pyqtSignal()

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def run(self):
        """Creates the dataset from captured images."""
        self.progress.emit("Iniciando creación del dataset...")
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        data = []
        labels = []
        
        # Check if data directory exists and has content
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            self.progress.emit("Error: Directorio de datos vacío o no encontrado. Por favor, capture imágenes primero.")
            self.finished.emit()
            return

        for dir_name in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, dir_name)
            if os.path.isdir(dir_path):
                self.progress.emit(f"Procesando clase: {dir_name}")
                for img_name in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_name)
                    
                    data_aux = []
                    x_ = []
                    y_ = []

                    img = cv2.imread(img_path)
                    if img is None:
                        self.progress.emit(f"Advertencia: No se pudo cargar la imagen {img_name}. Saltando.")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = hands.process(img_rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                data_aux.append(x - min(x_))
                                data_aux.append(y - min(y_))
                        
                        if len(data_aux) == 42: # 21 landmarks * 2 coordinates (x,y)
                            data.append(data_aux)
                            labels.append(dir_name)
                    else:
                        self.progress.emit(f"Advertencia: No se detectaron landmarks en {img_name}. Saltando.")

        try:
            with open('data.pickle', 'wb') as f:
                pickle.dump({'data': data, 'labels': labels}, f)
            self.progress.emit("Dataset creado exitosamente en data.pickle")
            self.dataset_created.emit()
        except Exception as e:
            self.progress.emit(f"Error al guardar data.pickle: {e}")
        finally:
            self.finished.emit()

class TrainModelWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    model_trained = pyqtSignal()

    def run(self):
        """Trains the model using the created dataset."""
        self.progress.emit("Iniciando entrenamiento del modelo...")
        try:
            data_dict = pickle.load(open('data.pickle', 'rb'))
        except FileNotFoundError:
            self.progress.emit("Error: data.pickle no encontrado. Por favor, cree el dataset primero.")
            self.finished.emit()
            return
        except Exception as e:
            self.progress.emit(f"Error al cargar data.pickle: {e}")
            self.finished.emit()
            return

        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        if len(data) == 0:
            self.progress.emit("Error: No hay datos para entrenar. El dataset está vacío.")
            self.finished.emit()
            return

        # Dividir los datos en conjuntos de entrenamiento y prueba
        try:
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        except ValueError as e:
            self.progress.emit(f"Error al dividir los datos: {e}. Asegúrese de tener al menos dos clases y suficientes muestras por clase.")
            self.finished.emit()
            return

        # Entrenar el modelo
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # Evaluar el modelo
        y_predict = model.predict(x_test)

        accuracy = accuracy_score(y_predict, y_test)
        precision = precision_score(y_predict, y_test, average='weighted', zero_division=0)
        recall = recall_score(y_predict, y_test, average='weighted', zero_division=0)
        f1 = f1_score(y_predict, y_test, average='weighted', zero_division=0)

        self.progress.emit(f'Accuracy: {accuracy:.4f}')
        self.progress.emit(f'Precision: {precision:.4f}')
        self.progress.emit(f'Recall: {recall:.4f}')
        self.progress.emit(f'F1 Score: {f1:.4f}')
        self.progress.emit(f'Confusion Matrix:\n{confusion_matrix(y_test, y_predict)}')

        # Guardar el modelo entrenado
        try:
            with open('model.p', 'wb') as f:
                pickle.dump({'model': model}, f)
            self.progress.emit("Modelo entrenado y guardado exitosamente en model.p")
            self.model_trained.emit()
        except Exception as e:
            self.progress.emit(f"Error al guardar model.p: {e}")
        finally:
            self.finished.emit()

class AdminViewModel(QObject):
    """
    ViewModel for the Admin View. Contains the business logic for
    data capture, dataset creation, and model training.
    """
    # Signals to update the UI
    enable_create_dataset_button = pyqtSignal(bool)
    enable_train_model_button = pyqtSignal(bool)
    log_message = pyqtSignal(str)
    frame_ready = pyqtSignal(QImage) # New signal for video frames from CaptureWorker
    
    # Signals to control worker threads
    capture_started = pyqtSignal()
    training_started = pyqtSignal()
    
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.capture_thread = None
        self.capture_worker = None
        self.dataset_thread = None
        self.dataset_worker = None
        self.train_thread = None
        self.train_worker = None

    def start_capture(self, num_people, category):
        """Starts the data capture process in a separate thread."""
        self.capture_worker = CaptureWorker(num_people, category)
        self.capture_thread = QThread()
        
        self.capture_worker.moveToThread(self.capture_thread)
        
        # Connect signals and slots
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        
        self.capture_worker.progress.connect(self.log_message)
        self.capture_worker.capture_finished.connect(lambda: self.enable_create_dataset_button.emit(True))
        self.capture_worker.frame_ready.connect(self.frame_ready) # Connect new frame signal

        self.capture_thread.start()
        self.log_message.emit("Iniciando proceso de captura...")
        self.capture_started.emit()

    def cancel_capture(self):
        """Stops the data capture process."""
        if self.capture_worker:
            self.capture_worker.stop()
        self.log_message.emit("Cancelando captura...")

    def create_dataset(self):
        """Starts the dataset creation process in a separate thread."""
        self.dataset_worker = DatasetWorker(data_dir="./data") # Assuming 'data' is the dir
        self.dataset_thread = QThread()

        self.dataset_worker.moveToThread(self.dataset_thread)

        # Connect signals and slots
        self.dataset_thread.started.connect(self.dataset_worker.run)
        self.dataset_worker.finished.connect(self.dataset_thread.quit)
        self.dataset_worker.finished.connect(self.dataset_worker.deleteLater)
        self.dataset_thread.finished.connect(self.dataset_thread.deleteLater)

        self.dataset_worker.progress.connect(self.log_message)
        self.dataset_worker.dataset_created.connect(lambda: self.enable_train_model_button.emit(True))

        self.dataset_thread.start()
        self.log_message.emit("Iniciando proceso de creación de dataset...")

    def train_model(self):
        """Starts the model training process in a separate thread."""
        self.train_worker = TrainModelWorker()
        self.train_thread = QThread()

        self.train_worker.moveToThread(self.train_thread)

        # Connect signals and slots
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.finished.connect(self.train_thread.quit)
        self.train_worker.finished.connect(self.train_worker.deleteLater)
        self.train_thread.finished.connect(self.train_thread.deleteLater)

        self.train_worker.progress.connect(self.log_message)
        self.train_worker.model_trained.connect(lambda: self.log_message.emit("Entrenamiento completado. Modelo listo para usar."))

        self.train_thread.start()
        self.log_message.emit("Iniciando proceso de entrenamiento de modelo...")
        self.training_started.emit()