# src/validate.py adaptado a raíz del proyecto
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys, os

THRESHOLD = 5000.0

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_path = os.path.abspath(os.path.join(os.getcwd(), "model.pkl"))
print(f"Intentando cargar modelo: {model_path}")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print("❌ No se encontró model.pkl. ¿Ejecutaste `make train`?", file=sys.stderr)
    sys.exit(1)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE: {mse:.4f} (umbral {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ Validación aprobada.")
    sys.exit(0)
else:
    print("❌ Validación fallida: supera el umbral.")
    sys.exit(1)