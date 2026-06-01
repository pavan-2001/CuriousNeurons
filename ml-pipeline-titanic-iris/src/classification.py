import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score


def build_pipeline(estimator, num_cols, cat_cols):
    """Generic preprocessing + model pipeline"""
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy='median')),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy='most_frequent')),
            ("oh", OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    return Pipeline([("pre", pre), ("model", estimator)])

def evaluate(pipe, X, Y, cv, scoring, task):
    cvr = cross_validate(pipe, X, Y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    print(F"\n=== {task} CV scores ===")
    for s in scoring:
        v = cvr[f"test_{s}"]
        print(f"    {s:25s} {v.mean():.4f} ± {v.std():.4f}")


# 1. Classification: Titanic
titanic = pd.read_csv('./ml-pipeline-titanic-iris/data/titanic/train.csv')
Y_c = titanic['Survived'].astype(int)
X_c = titanic.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], errors='ignore')
num_c = ['Age', 'SibSp', 'Parch', 'Fare']
cat_c = ['Pclass', 'Sex', 'Embarked']

clf = build_pipeline(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), num_c, cat_c)
evaluate(clf, X_c, Y_c, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring=['accuracy', 'f1', 'roc_auc'], task='Titanic Classification')

Xtr, Xv, Ytr, Yv = train_test_split(X_c, Y_c, test_size=.2, stratify=Y_c, random_state=0)
clf.fit(Xtr, Ytr)

print(classification_report(Yv, clf.predict(Xv)))

# Predictions for Kaggle submission for titanic

# Xte = pd.read_csv('./ml-pipeline-titanic-iris/data/titanic/test.csv')
# passenger_ids = Xte['PassengerId'] if 'PassengerId' in Xte.columns else None

# Xte = Xte.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], errors='ignore')

# Yp = clf.predict(Xte)

# result = pd.DataFrame({
#     'PassengerId': passenger_ids,
#     'Survived': Yp
# })

# result.to_csv('./random_forest_output.csv', index=False)