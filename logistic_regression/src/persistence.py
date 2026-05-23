import os
import joblib

ARTIFACT_PATH = "artifacts/titanic_model.joblib"

def save_artifact(model, preprocessor, path=ARTIFACT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'preprocessor': preprocessor}, path)
    print(f"Saved artifact to {path}")

def load_artifact(path=ARTIFACT_PATH):
    artifact = joblib.load(path)
    return artifact['model'], artifact['preprocessor']