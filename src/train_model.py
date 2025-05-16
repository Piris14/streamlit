
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from pickle import dump
import os

# Cargar datos iris
iris = load_iris()
X, y = iris.data, iris.target

# Entrenar modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Guardar modelo entrenado
model_path = "models/decision_tree_classifier_default_42.sav"
with open(model_path, "wb") as f:
    dump(model, f)

print(f"Modelo guardado en {model_path}")
