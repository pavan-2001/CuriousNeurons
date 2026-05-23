import pandas as pd
from .persistence import load_artifact
from .metrics import accuracy, precision, recall

def predict_from_csv(csv_path, output_path="predictions.csv"):
    df = pd.read_csv(csv_path)

    passenger_ids = df['PassengerId'] if 'PassengerId' in df.columns else None
    model, preprocessor = load_artifact()

    X, Y = preprocessor.transform(df)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    result.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    return result

def df_for_inference(df):
    if 'Survived' not in df.columns:
        df = df.copy()
        df['Survived'] = 0
    return df