import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3  
dataset_size = 100  

# Diccionario para asociar el número de clase con señas
labels_dict = {
    0: 'Peligro',
    1: 'Todo_OK',
    2: 'Ninguna_Sena'
}

# Preguntar cuántas personas se utilizarán
num_people = int(input('¿Cuántas personas van a participar en la captura de datos? '))

for person_idx in range(num_people):
    print(f'\nPreparando para la persona {person_idx + 1}...')
    for class_num in range(number_of_classes):
        class_name = labels_dict[class_num]
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        print(f'Recolectando datos para la clase: {class_name}')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            exit()

        done_collecting = False
        while not done_collecting:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                break

            cv2.putText(frame, 'Listo? Presiona "Q" para empezar!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                done_collecting = True

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            img_path = os.path.join(class_path, f'{int(time.time() * 1000)}.jpg')
            cv2.imwrite(img_path, frame)

            counter += 1
            print(f'Capturando imagen {counter}/{dataset_size} para {class_name}')

        cap.release()
        cv2.destroyAllWindows()

print("Captura de datos completada.")
