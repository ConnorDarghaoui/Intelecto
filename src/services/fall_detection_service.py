import os
import sys
import paddle
import numpy as np
from typing import Optional, Dict, Any, List

# Agregar pipeline al path
pipeline_path = os.path.join(os.path.dirname(__file__), "../../pipeline")
sys.path.insert(0, pipeline_path)

try:
    from pipeline import Pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pipeline module not found: {e}. Fall detection will be disabled.")
    Pipeline = None
    PIPELINE_AVAILABLE = False


def get_inference_device() -> str:
    """
    Detecta automáticamente el dispositivo de inferencia disponible
    
    Returns:
        str: 'gpu' si está disponible, 'cpu' en caso contrario
    """
    try:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
        else:
            return "cpu"
    except Exception as e:
        print(f"Error detecting device: {e}")
        return "cpu"


class FallDetectionService:
    """
    Servicio de detección de caídas usando PaddleDetection Pipeline
    Integrado con la arquitectura MVVM del proyecto
    """
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.8):
        """
        Inicializa el servicio de detección de caídas
        
        Args:
            device: Dispositivo de inferencia ('cpu' o 'gpu'). Si es None, se detecta automáticamente
            confidence_threshold: Umbral de confianza para detección de caídas
        """
        self.device = device or get_inference_device()
        self.confidence_threshold = confidence_threshold
        self.pipeline = None
        self.enabled = False
        
        print(f"[FallDetectionService] Inicializando en dispositivo: {self.device.upper()}")
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Inicializa el pipeline de PaddleDetection"""
        if not PIPELINE_AVAILABLE:
            print("[FallDetectionService] Pipeline no disponible - detección de caídas deshabilitada")
            return
        
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "../../pipeline/config/infer_cfg_pphuman.yml"
            )
            
            if not os.path.exists(config_path):
                print(f"[FallDetectionService] Config no encontrado: {config_path}")
                return
            
            self.pipeline = Pipeline(config_path=config_path, device=self.device)
            self.enabled = True
            print("[FallDetectionService] Pipeline inicializado correctamente")
            
        except Exception as e:
            print(f"[FallDetectionService] Error inicializando pipeline: {e}")
            self.enabled = False
    
    def detect_fall(self, frame: np.ndarray) -> bool:
        """
        Detecta caídas en un frame de video
        
        Args:
            frame: Frame de video (numpy array BGR)
            
        Returns:
            bool: True si se detecta una caída, False en caso contrario
        """
        if not self.enabled or self.pipeline is None:
            return False
        
        try:
            # Ejecutar pipeline de detección
            outputs = self.pipeline.run(frame)
            
            # Buscar detecciones de caídas
            for detection in outputs:
                if (detection.get("category_id") == 0 and  # 0 = fall category
                    detection.get("score", 0) > self.confidence_threshold):
                    return True
            
            return False
            
        except Exception as e:
            print(f"[FallDetectionService] Error en detección: {e}")
            return False
    
    def get_detection_details(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Obtiene detalles completos de las detecciones
        
        Args:
            frame: Frame de video (numpy array BGR)
            
        Returns:
            List[Dict]: Lista de detecciones con detalles
        """
        if not self.enabled or self.pipeline is None:
            return []
        
        try:
            outputs = self.pipeline.run(frame)
            
            fall_detections = []
            for detection in outputs:
                if detection.get("category_id") == 0:  # fall category
                    fall_detections.append({
                        "category": "fall",
                        "confidence": detection.get("score", 0),
                        "bbox": detection.get("bbox", []),
                        "timestamp": None  # Se puede agregar timestamp
                    })
            
            return fall_detections
            
        except Exception as e:
            print(f"[FallDetectionService] Error obteniendo detalles: {e}")
            return []
    
    def is_enabled(self) -> bool:
        """Verifica si el servicio está habilitado y funcionando"""
        return self.enabled
    
    def get_device_info(self) -> Dict[str, Any]:
        """Obtiene información del dispositivo de inferencia"""
        try:
            cuda_available = paddle.is_compiled_with_cuda() if paddle else False
            gpu_count = paddle.device.cuda.device_count() if paddle and cuda_available else 0
        except:
            cuda_available = False
            gpu_count = 0
            
        return {
            "device": self.device,
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count
        }


# Función de utilidad para testing
def test_fall_detection_service():
    """Prueba básica del servicio de detección de caídas"""
    service = FallDetectionService()
    
    print("=== Test Fall Detection Service ===")
    print(f"Enabled: {service.is_enabled()}")
    print(f"Device info: {service.get_device_info()}")
    
    # Test con frame dummy
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = service.detect_fall(dummy_frame)
    print(f"Fall detection result: {result}")
    
    details = service.get_detection_details(dummy_frame)
    print(f"Detection details: {details}")


if __name__ == "__main__":
    test_fall_detection_service()
