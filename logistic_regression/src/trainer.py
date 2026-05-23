from .data_loader import load_train_data
from .preprocessing import TitanicPreprocessor, train_test_split
from .model import LogisticRegression
from .metrics import accuracy, precision, recall
from .persistence import save_artifact

def run_pipeline():
    df = load_train_data()

    train_df, valid_df = train_test_split(df)
    preprocessor = TitanicPreprocessor()

    preprocessor.fit(train_df)

    X_train, Y_train = preprocessor.transform(train_df)
    X_test, Y_test = preprocessor.transform(valid_df)

    

    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, Y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("Train Accuracy:", accuracy(Y_train, train_preds))
    print("Test Accuracy: ", accuracy(Y_test, test_preds))
    print("Test Precision: ", precision(Y_test, test_preds))
    print("Test Recall: ", recall(Y_test, test_preds))

    save_artifact(model ,preprocessor)

    return model