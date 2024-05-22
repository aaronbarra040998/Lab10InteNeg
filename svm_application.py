import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Leer datos del archivo Data10.xlsx
data = pd.read_excel('Data10.xlsx')

# Imprimir los nombres de las columnas para verificación
print(data.columns)

# Usar 'Precio actual' y 'Precio final' como características y 'Estado' como etiqueta
X = data[['Precio actual', 'Precio final']].values
y = data['Estado'].values

# Convertir etiquetas de texto a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear el modelo SVM
svm_model = SVC(kernel='linear')

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred = svm_model.predict(X_test)

# Evaluar el modelo
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}')

# Visualizar el hiperplano de separación (para 2 características)
def plot_svm_boundary(model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

# Llamar a la función para graficar
plot_svm_boundary(svm_model, X_train, y_train)
