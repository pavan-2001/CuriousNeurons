import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def build_pipeline(estimator, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        (
            "cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols
        )
    ])

    return Pipeline([("pre", pre), ("model", estimator)])

def evaluate(pipe, X, Y, cv, scoring, task):
    cvr = cross_validate(pipe, X, Y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    print(f"\n{task} CV score")
    for s in scoring:
        v = cvr[f"test_{s}"]
        print(f"    {s:25s} {v.mean():.4f} +- {v.std():.4f}")

iris = pd.read_csv('./ml-pipeline-titanic-iris/data/iris/iris_train.csv')
iris = iris.dropna(subset='petal_length')
Y_r = iris['petal_length']
X_r = iris.drop(columns=['petal_length'], errors='ignore')

num_r = [
    'sepal_length', 'sepal_width', 'petal_width'
]

cat_r = ['species']

reg = build_pipeline(RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1), num_r, cat_r)
evaluate(reg, X_r, Y_r,
    cv=KFold(5, shuffle=True, random_state=0),
    scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
    task="Iris Regression"
)

Xtr, X_v, Ytr, Y_v = train_test_split(X_r, Y_r, test_size=0.2, random_state=0)
reg.fit(Xtr, Ytr)
pred = reg.predict(X_v)

print(f"Holdout R2={r2_score(Y_v, pred):.4f} RMSE={mean_squared_error(Y_v, pred)**.5:.4f}")