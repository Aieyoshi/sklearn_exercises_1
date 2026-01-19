# sklearn_exercises_1

Práctica básica de **scikit-learn** usando un **DecisionTreeClassifier**:
- Generación de un dataset **sintético** (`generate_dataset.py`)
- Creación de una etiqueta binaria de precio (`price_class`)
- Entrenamiento con `fit()`
- Predicción de nuevos ejemplos con `predict()`

---

## Estructura del proyecto

- `generate_dataset.py` → genera `datasets/train_data_us.csv`
- `exercise_predict.py` → entrena el modelo y predice 2 apartamentos nuevos
- `datasets/train_data_us.csv` → dataset sintético (features + `last_price`)

---

## Requisitos

- Python (en Windows suele funcionar con `py`)
- Librerías:
  - `pandas`
  - `scikit-learn`