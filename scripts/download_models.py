#!/usr/bin/env python3
"""
Script para descargar modelos de PaddleDetection necesarios para detección de caídas.
Descarga automáticamente los 3 modelos principales:
1. MOT (Multi-Object Tracking)
2. Keypoint Detection 
3. STGCN (Spatio-Temporal Graph Convolutional Network)
"""

import os
import urllib.request
import zipfile
import sys
from pathlib import Path

# URLs de los modelos según las instrucciones
MODEL_URLS = {
    "mot_ppyoloe_l_36e_pipeline.zip": "https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip",
    "dark_hrnet_w32_256x192.zip": "https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip", 
    "STGCN.zip": "https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip"
}

def download_file(url: str, destination: str) -> bool:
    """Descarga un archivo desde una URL"""
    try:
        print(f"Descargando {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Descarga completada: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extrae un archivo ZIP"""
    try:
        print(f"Extrayendo {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracción completada: {zip_path}")
        return True
    except Exception as e:
        print(f"❌ Error extrayendo {zip_path}: {e}")
        return False

def main():
    """Función principal para descargar y extraer modelos"""
    # Crear directorio de destino
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "pipeline" / "output_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Descargando modelos de PaddleDetection para detección de caídas...")
    print(f"📁 Directorio de destino: {output_dir}")
    
    success_count = 0
    total_models = len(MODEL_URLS)
    
    for filename, url in MODEL_URLS.items():
        zip_path = output_dir / filename
        
        # Verificar si ya existe
        if zip_path.exists():
            print(f"⏭️ {filename} ya existe, saltando descarga...")
            success_count += 1
            continue
        
        # Descargar archivo
        if download_file(url, str(zip_path)):
            # Extraer archivo
            extract_dir = output_dir / filename.replace('.zip', '')
            if extract_zip(str(zip_path), str(extract_dir)):
                # Eliminar ZIP después de extraer
                zip_path.unlink()
                print(f"🗑️ Archivo ZIP eliminado: {filename}")
                success_count += 1
            else:
                print(f"⚠️ Error extrayendo {filename}, manteniendo ZIP")
        
        print("-" * 50)
    
    # Resumen final
    print("\n" + "="*60)
    print(f"📊 RESUMEN: {success_count}/{total_models} modelos procesados correctamente")
    
    if success_count == total_models:
        print("🎉 ¡Todos los modelos descargados exitosamente!")
        print("\n📋 Modelos disponibles:")
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                print(f"   ✅ {model_dir.name}")
        
        print(f"\n🔧 Configuración:")
        print(f"   📁 Pipeline config: {base_dir}/pipeline/config/infer_cfg_pphuman.yml")
        print(f"   📁 Modelos en: {output_dir}")
        
        print("\n🚀 ¡Ya puedes usar la detección de caídas!")
        
    else:
        print("⚠️ Algunos modelos no se descargaron correctamente.")
        print("Por favor, revisa los errores arriba e intenta nuevamente.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
