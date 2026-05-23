from src.trainer import run_pipeline
import sys
from src.predict import predict_from_csv

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_from_csv("data/test.csv")
    else:
        run_pipeline()
