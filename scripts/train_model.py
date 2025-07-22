import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Cargar el dataset
data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluar el modelo
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_predict, y_test, average='weighted', zero_division=0)
recall = recall_score(y_predict, y_test, average='weighted', zero_division=0)
f1 = f1_score(y_predict, y_test, average='weighted', zero_division=0)

cm = confusion_matrix(y_test, y_predict)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{cm}')

# Guardar el modelo entrenado
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print("Modelo entrenado y guardado exitosamente en model.p")
