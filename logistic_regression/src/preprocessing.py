import numpy as np
import pandas as pd

class TitanicPreprocessor:
    def __init__(self):
        self.age_median = None
        self.embarked_mode = None
        self.means = None
        self.stds = None
        self.feature_columns = None
        self.fare_median = None

    def fit(self, df):
        df = df.copy()
        self.age_median = df['Age'].median()
        self.embarked_mode = df['Embarked'].mode()[0]
        self.fare_median = df['Fare'].median()

        features = self._prepare_features(df)

        self.feature_columns = features.columns
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0)

        return self
    
    def transform(self, df):
        df = df.copy()

        Y = df['Survived'].values if 'Survived' in df.columns else None

        features = self._prepare_features(df)

        features = features.reindex(columns=self.feature_columns, fill_value=0)

        features = (features - self.means) / self.stds

        return features.values, Y

    def _prepare_features(self, df):
        df = df.copy()

        df['Age'] = df['Age'].fillna(self.age_median)
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        df['Fare'] = df['Fare'].fillna(self.fare_median)

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']

        df['Title'] = df['Name'].str.extract(r",\s*([^\.]+)\.", expand=False)
        df['Title'] = df['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            "Rare"
        )
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

        df['AgeBucket'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 60, np.inf],
            labels=['child', 'Teen', 'YoungAdult', 'Adult', 'Senior'],
            include_lowest=True
        )

        df['FareBucket'] = pd.cut(
            df['Fare'],
            bins=[-np.inf, 8, 15, 31, np.inf],
            labels = ['Low', 'Mid', 'High', 'VeryHigh']
        )

        drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']

        if 'Survived' in df.columns:
            drop_columns.append('Survived')

        df = df.drop(columns=drop_columns, errors='ignore')

        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df = pd.get_dummies(df, columns=['Embarked', 'Title', 'AgeBucket', 'FareBucket'], dtype=int)

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
    
def train_test_split(df, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(df.shape[0])
    split = int((1 - test_size) * df.shape[0])

    train_df = df.iloc[indices[:split]].copy()
    valid_df = df.iloc[indices[split:]].copy()

    print(f"train_df: {train_df.shape}, valid_df: {valid_df.shape}")

    return train_df, valid_df