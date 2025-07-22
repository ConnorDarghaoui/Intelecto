# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import cv2
import numpy as np
import sys
from typing import List, Dict, Any


class Pipeline(object):
    """
    Simplified Pipeline for fall detection
    
    Args:
        config_path (str): path to config file
        device (str): device type, 'cpu' or 'gpu'
    """

    def __init__(self, config_path, device='cpu'):
        self.config_path = config_path
        self.device = device
        self.enabled = False
        
        print(f"[Pipeline] Inicializando pipeline en: {device}")
        
        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.cfg = yaml.safe_load(f)
            print(f"[Pipeline] Configuración cargada desde: {config_path}")
        except Exception as e:
            print(f"[Pipeline] Error cargando config: {e}")
            self.cfg = {}
        
        # Check if models exist
        self._check_models()

    def _check_models(self):
        """Check if required models exist"""
        output_dir = os.path.join(os.path.dirname(self.config_path), "../output_inference")
        
        required_models = [
            "mot_ppyoloe_l_36e_pipeline",
            "dark_hrnet_w32_256x192", 
            "STGCN"
        ]
        
        models_found = 0
        for model in required_models:
            model_path = os.path.join(output_dir, model)
            if os.path.exists(model_path):
                models_found += 1
                print(f"[Pipeline] ✅ Modelo encontrado: {model}")
            else:
                print(f"[Pipeline] ❌ Modelo faltante: {model}")
        
        if models_found == len(required_models):
            self.enabled = True
            print(f"[Pipeline] ✅ Pipeline habilitado - {models_found}/{len(required_models)} modelos disponibles")
        else:
            print(f"[Pipeline] ⚠️ Pipeline deshabilitado - solo {models_found}/{len(required_models)} modelos disponibles")

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run pipeline on a single frame
        
        Args:
            frame: input frame (numpy array)
            
        Returns:
            List of detection results
        """
        if not self.enabled:
            return []
        
        # Simplified fall detection logic
        # In a real implementation, this would involve:
        # 1. MOT (Multi-Object Tracking) to detect and track people
        # 2. Keypoint detection to get human pose
        # 3. STGCN to analyze temporal motion patterns
        
        results = []
        
        # Mock fall detection based on basic image analysis
        fall_detected = self._simple_fall_detection(frame)
        
        if fall_detected:
            results.append({
                'category_id': 0,  # 0 for fall
                'score': 0.85,     # mock confidence
                'bbox': [100, 100, 200, 200]  # mock bounding box
            })
        
        return results
    
    def _simple_fall_detection(self, frame: np.ndarray) -> bool:
        """
        Simplified fall detection based on basic image analysis
        This is a placeholder for the actual PaddleDetection pipeline
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic motion detection (very simplified)
            # In reality, this would use the full MOT+Keypoint+STGCN pipeline
            
            # Calculate image statistics
            height, width = gray.shape
            mean_intensity = np.mean(gray)
            
            # Mock fall detection logic - this is just for demonstration
            # Real implementation would analyze human poses and movement patterns
            
            # For demonstration, randomly trigger fall detection very rarely
            import random
            if random.random() < 0.001:  # 0.1% chance for demo
                return True
            
            return False
            
        except Exception as e:
            print(f"[Pipeline] Error en detección: {e}")
            return False


def get_inference_device():
    """Check if GPU is available for inference"""
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
        else:
            return "cpu"
    except:
        return "cpu"


if __name__ == '__main__':
    # Example usage
    config_path = "config/infer_cfg_pphuman.yml"
    device = get_inference_device()
    
    pipeline = Pipeline(config_path, device)
    
    # Test with dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = pipeline.run(dummy_frame)
    print("Detection results:", results)
