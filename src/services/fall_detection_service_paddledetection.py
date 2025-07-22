import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import os
import sys

# Agregar el directorio del pipeline al path
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(current_dir, '../../pipeline')
sys.path.append(pipeline_dir)

class IntegratedDetectionService:
    """
    Servicio integrado que combina:
    1. MediaPipe Hands (manos - puntos verdes)
    2. PaddleHub Pose (esqueleto completo - puntos azules)
    3. PaddleDetection Falling (detección de caídas - cajas y etiquetas rojas)
    """
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.8):
        """
        Inicializa el servicio integrado de detección
        """
        self.device = device or "cpu"
        self.confidence_threshold = confidence_threshold
        self.enabled = False
        
        # Inicializar MediaPipe Hands
        self._init_mediapipe_hands()
        
        # Inicializar PaddleHub Pose
        self._init_paddlehub_pose()
        
        # Inicializar PaddleDetection Falling Pipeline
        self._init_falling_pipeline()
        
        # Conexiones del esqueleto COCO (17 keypoints)
        self.SKELETON_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),        # cara
            (5,6),(5,7),(7,9),(6,8),(8,10), # brazos
            (11,12),(11,13),(13,15),(12,14),(14,16), # piernas
            (11,23),(12,24)                 # cadera al torso
        ]
        
    def _init_mediapipe_hands(self):
        """Inicializar MediaPipe Hands"""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands_proc = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[IntegratedService] ✅ MediaPipe Hands inicializado")
        except Exception as e:
            print(f"[IntegratedService] ❌ Error inicializando MediaPipe Hands: {e}")
            self.hands_proc = None
            
    def _init_paddlehub_pose(self):
        """Inicializar MediaPipe Pose (mejor compatibilidad que PaddleHub)"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_processor = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("[IntegratedService] ✅ MediaPipe Pose inicializado (reemplazando PaddleHub)")
        except Exception as e:
            print(f"[IntegratedService] ❌ Error inicializando MediaPipe Pose: {e}")
            self.pose_processor = None
            
    def _init_falling_pipeline(self):
        """Inicializar PaddleDetection Falling Pipeline"""
        try:
            from pipeline import Pipeline
            
            config_file = os.path.join(pipeline_dir, 'config/infer_cfg_pphuman.yml')
            
            if os.path.exists(config_file):
                self.fall_pipeline = Pipeline(
                    config_path=config_file,
                    device=self.device
                )
                self.enabled = True
                print(f"[IntegratedService] ✅ PaddleDetection Falling inicializado en: {self.device.upper()}")
            else:
                print(f"[IntegratedService] ❌ Config no encontrado: {config_file}")
                self.fall_pipeline = None
                
        except Exception as e:
            print(f"[IntegratedService] ❌ Error inicializando PaddleDetection: {e}")
            self.fall_pipeline = None
            
    def draw_hands(self, img: np.ndarray, hands_res) -> None:
        """Dibuja landmarks de mano en verde"""
        if hands_res and hands_res.multi_hand_landmarks:
            h, w = img.shape[:2]
            for hl in hands_res.multi_hand_landmarks:
                for lm in hl.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Verde
                    
    def draw_pose(self, img: np.ndarray, pose_res) -> None:
        """Dibuja esqueleto completo en azul usando MediaPipe"""
        if not pose_res or not pose_res.pose_landmarks:
            return
            
        try:
            # Dibujar conexiones y landmarks del esqueleto
            self.mp_drawing.draw_landmarks(
                img,
                pose_res.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=4, circle_radius=4  # Azul
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=3  # Azul
                )
            )
                            
        except Exception as e:
            print(f"[IntegratedService] Error dibujando pose: {e}")
            
    def draw_falls(self, img: np.ndarray, falls: List[Dict]) -> bool:
        """Dibuja cajas y etiquetas de caída en rojo/verde"""
        fall_detected = False
        
        for idx, det in enumerate(falls):
            try:
                x1, y1, x2, y2 = map(int, det['bbox'])
                score = det['score']
                pid = det.get('id', idx)
                
                # Caja verde para persona detectada
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"ID:{pid} {score:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)
                
                # Etiqueta roja para caída
                if det.get('category_id') == 0 and score > self.confidence_threshold:
                    cv2.putText(img, "Falling",
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)  # Rojo
                    fall_detected = True
                    
            except Exception as e:
                print(f"[IntegratedService] Error dibujando caída {idx}: {e}")
                
        return fall_detected
        
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Procesa un frame con los tres modelos integrados
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            bool: True si se detecta una caída, False en caso contrario
        """
        if not self.enabled or frame is None or frame.size == 0:
            return False
            
        fall_detected = False
        
        try:
            # A) Detectar y dibujar manos con MediaPipe
            if self.hands_proc:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands_res = self.hands_proc.process(rgb)
                self.draw_hands(frame, hands_res)
                
                # Mostrar estado de manos
                if hands_res and hands_res.multi_hand_landmarks:
                    cv2.putText(frame, "Manos detectadas", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No se detectan manos", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # B) Detectar y dibujar pose completa con MediaPipe
            if self.pose_processor:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_res = self.pose_processor.process(rgb)
                self.draw_pose(frame, pose_res)
                
                # Mostrar estado de pose
                if pose_res and pose_res.pose_landmarks:
                    cv2.putText(frame, "Esqueleto detectado", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Pose tu cuerpo frente a la camara", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # C) Detectar y dibujar caídas con PaddleDetection
            if self.fall_pipeline:
                fall_res = self.fall_pipeline.run(frame)
                fall_detected = self.draw_falls(frame, fall_res)
                
                # Mostrar estado de servicio
                cv2.putText(frame, "Pipeline integrado activo", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                          
                if fall_detected:
                    cv2.putText(frame, "¡CAÍDA DETECTADA!", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
        except Exception as e:
            print(f"[IntegratedService] Error procesando frame: {e}")
            cv2.putText(frame, "Error en pipeline integrado", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        return fall_detected
        
    def is_enabled(self) -> bool:
        """Retorna si el servicio está habilitado"""
        return self.enabled
        
    def get_detection_details(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Obtiene detalles de la detección integrada
        """
        if not self.enabled or self.fall_pipeline is None:
            return []
            
        try:
            results = self.fall_pipeline.run(frame)
            details = []
            
            for result in results:
                if isinstance(result, dict):
                    details.append({
                        'action': result.get('action', 'detected'),
                        'confidence': result.get('score', 0.0),
                        'bbox': result.get('bbox', []),
                        'category_id': result.get('category_id', -1)
                    })
                    
            return details
            
        except Exception as e:
            print(f"[IntegratedService] Error obteniendo detalles: {e}")
            return []

# Wrapper para mantener compatibilidad con el código existente
class FallDetectionService:
    """
    Wrapper de compatibilidad para IntegratedDetectionService
    Mantiene la misma interfaz que el código existente
    """
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.8):
        self.integrated_service = IntegratedDetectionService(device, confidence_threshold)
        
    def detect_fall(self, frame: np.ndarray) -> bool:
        """Detecta caídas usando el servicio integrado"""
        return self.integrated_service.process_frame(frame)
        
    def is_enabled(self) -> bool:
        """Retorna si el servicio está habilitado"""
        return self.integrated_service.is_enabled()
        
    def get_detection_details(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Obtiene detalles de la detección"""
        return self.integrated_service.get_detection_details(frame)
