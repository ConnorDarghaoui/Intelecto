#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de los tres modelos:
1. MediaPipe Hands
2. PaddleHub Pose  
3. PaddleDetection Falling

Ejecuta este script para probar cada modelo por separado.
"""

import cv2
import numpy as np

def test_mediapipe_hands():
    """Prueba MediaPipe Hands"""
    print("🧪 Probando MediaPipe Hands...")
    try:
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Hands: OK")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe Hands: {e}")
        return False

def test_mediapipe_pose():
    """Prueba MediaPipe Pose (reemplaza PaddleHub)"""
    print("🧪 Probando MediaPipe Pose...")
    try:
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Pose: OK")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe Pose: {e}")
        return False

def test_paddledetection_falling():
    """Prueba PaddleDetection Falling Pipeline"""
    print("🧪 Probando PaddleDetection Falling...")
    try:
        import sys
        import os
        
        # Agregar pipeline al path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_dir = os.path.join(current_dir, 'pipeline')
        sys.path.append(pipeline_dir)
        
        from pipeline import Pipeline
        
        config_file = os.path.join(pipeline_dir, 'config/infer_cfg_pphuman.yml')
        
        if os.path.exists(config_file):
            fall_pipeline = Pipeline(
                config_path=config_file,
                device="cpu"  # Usar CPU para la prueba
            )
            print("✅ PaddleDetection Falling: OK")
            return True
        else:
            print(f"❌ PaddleDetection Falling: Config no encontrado: {config_file}")
            return False
            
    except Exception as e:
        print(f"❌ PaddleDetection Falling: {e}")
        return False

def test_integrated_service():
    """Prueba el servicio integrado"""
    print("🧪 Probando servicio integrado...")
    try:
        from src.services.fall_detection_service_paddledetection import IntegratedDetectionService
        
        service = IntegratedDetectionService(device="cpu")
        
        if service.is_enabled():
            print("✅ Servicio integrado: OK")
            return True
        else:
            print("⚠️ Servicio integrado: Parcialmente disponible")
            return False
            
    except Exception as e:
        print(f"❌ Servicio integrado: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("🚀 Iniciando pruebas de integración...")
    print("=" * 50)
    
    tests = [
        ("MediaPipe Hands", test_mediapipe_hands),
        ("MediaPipe Pose", test_mediapipe_pose),
        ("PaddleDetection Falling", test_paddledetection_falling),
        ("Servicio Integrado", test_integrated_service)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 {name}")
        success = test_func()
        results.append((name, success))
        print("-" * 30)
    
    print("\n📊 RESUMEN DE PRUEBAS:")
    print("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<25} {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ¡Todas las pruebas pasaron! La integración está lista.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa las dependencias.")

if __name__ == "__main__":
    main()
