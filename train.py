import os, sys, traceback, joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Rutas y tracking local ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
mlflow.set_tracking_uri(tracking_uri)

# --- Experimento explícito ---
EXPERIMENT_NAME = "CI-CD-Lab2"
artifact_location = tracking_uri
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
except Exception:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = exp.experiment_id if exp else None
if experiment_id is None:
    print("No se pudo crear/obtener el experimento", file=sys.stderr)
    sys.exit(1)

# --- Entrenamiento ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Logging y guardado ---
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="model")
    joblib.dump(model, "model.pkl")            # ← para validate.py
    mlflow.log_artifact("model.pkl")
    print(f"✅ Entrenado. MSE={mse:.4f}")
    print(f"📁 MLflow tracking URI: {tracking_uri}")
    print(f"📦 Artifact URI: {run.info.artifact_uri}")