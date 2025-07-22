#!/usr/bin/env python3
"""
🎬 Demo de Integración - Intelecto
==================================

Demostración en tiempo real de los tres modelos trabajando juntos:
1. MediaPipe Hands (manos - verde)
2. MediaPipe Pose (esqueleto - azul) 
3. PaddleDetection Falling (caídas - rojo)

Usa tu cámara web para ver todos los modelos funcionando simultáneamente.
"""

import cv2
import sys
import os

# Agregar el directorio del proyecto al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.services.fall_detection_service_paddledetection import IntegratedDetectionService

def main():
    print("🎬 DEMO: Integración de Tres Modelos")
    print("=" * 50)
    print("📹 Iniciando cámara...")
    
    # Inicializar servicio integrado
    service = IntegratedDetectionService(device="cpu")
    
    if not service.is_enabled():
        print("❌ Error: El servicio integrado no está disponible")
        print("💡 Ejecuta: python test_integration.py")
        return
    
    print("✅ Servicio integrado inicializado correctamente")
    print()
    print("🎯 INSTRUCCIONES:")
    print("- Colócate frente a la cámara")
    print("- Mueve las manos para ver landmarks verdes")
    print("- El esqueleto aparece en azul") 
    print("- Las cajas verdes muestran personas detectadas")
    print("- Presiona 'ESC' para salir")
    print("- Presiona 'i' para información detallada")
    print()
    
    # Configurar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🚀 ¡Demo iniciado! Mira la ventana de video...")
    
    frame_count = 0
    info_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo de la cámara")
            break
        
        frame_count += 1
        
        # Procesar frame con servicio integrado
        fall_detected = service.process_frame(frame)
        
        # Agregar información de overlay
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Título principal
        cv2.putText(frame, "DEMO: Tres Modelos Integrados", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instrucciones
        if info_mode:
            instructions = [
                "MODELOS ACTIVOS:",
                "Verde: MediaPipe Hands (21 landmarks)",
                "Azul: MediaPipe Pose (17 keypoints)", 
                "Rojo: PaddleDetection Falling",
                "",
                "CONTROLES:",
                "ESC: Salir  |  i: Info"
            ]
            
            for i, text in enumerate(instructions):
                color = (0, 255, 255) if ":" in text else (255, 255, 255)
                cv2.putText(frame, text, 
                           (frame.shape[1] - 300, 50 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Mostrar alerta de caída si se detecta
        if fall_detected:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(frame, "¡CAIDA DETECTADA!", 
                       (frame.shape[1]//2 - 100, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Intelecto - Demo Integración", frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('i'):  # Toggle info
            info_mode = not info_mode
        elif key == ord('s'):  # Screenshot
            filename = f"demo_screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot guardado: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("👋 Demo finalizado")
    print("📊 Estadísticas:")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Modelos activos: 3/3")
    print("✨ ¡Gracias por probar Intelecto!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        print("💡 Verifica que la cámara esté disponible y los modelos instalados")
