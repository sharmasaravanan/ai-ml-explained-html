import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# 1. Load and inspect (Section 9 Hands-On Bridge)
df = pd.read_csv("customer_churn_preprocessing.csv")
print(df.info())
print("Duplicates:", df.duplicated().sum())

# 2. Clean text, duplicates, types, and impossible outliers BEFORE splitting
df["city"] = df["city"].str.strip().str.title()
df["city"] = df["city"].replace({"Chenai": "Chennai", "Bangalore": "Bengaluru"})
df = df.drop_duplicates()

df["age"] = pd.to_numeric(df["age"], errors="coerce")  # "unknown" & "N/A" -> NaN
df.loc[df["age"] > 100, "age"] = np.nan                # age 120 -> NaN

# 3. Split first to prevent data leakage (Section 1 & Section 6)
X = df[["city", "age", "size", "salary"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build leak-proof ColumnTransformer + Pipeline (Section 7)
num_cols = ["age", "salary"]
nom_cols = ["city"]
ord_cols = ["size"]

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
])

nom_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore")),
])

ord_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(categories=[["S", "M", "L", "XL"]])),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("nom", nom_pipe, nom_cols),
    ("ord", ord_pipe, ord_cols),
])

clf = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression()),
])

clf.fit(X_train, y_train)
print("Test accuracy:", clf.score(X_test, y_test))