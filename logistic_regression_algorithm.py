import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, ConfusionMatrixDisplay, \
    classification_report
from IPython.display import display
from warnings import filterwarnings
filterwarnings("ignore")

# 1. Load and inspect (Section 9 Hands-On Bridge)
df = pd.read_csv("customer_churn_preprocessing_extended.csv")
display(df.info())
# display(df.describe())
print(df.isnull().sum())
print(df.isna().sum())
print("Duplicates:", df.duplicated().sum())

# 2. Clean text, duplicates, types, and impossible outliers BEFORE splitting
df["city"] = df["city"].str.strip().str.title()
df["city"] = df["city"].replace({"Chenai": "Chennai", "Bangalore": "Bengaluru"})
df = df.drop_duplicates()

df["age"] = pd.to_numeric(df["age"], errors="coerce")  # "unknown" & "N/A" -> NaN
df.loc[df["age"] > 100, "age"] = np.nan                # age 120 -> NaN

# fill the null values with median
for col in df.columns:
    if df[col].dtypes != "object":
        df[col] = df[col].fillna(value=df[col].median())
    else:
        df[col] = df[col].fillna(value=df[col].mode()[0])

# 3. Split first to prevent data leakage (Section 1 & Section 6)
X = df[["city", "age", "size", "salary"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

print(df.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
display(X_train.head())

# converting City and Size into numerical column using label encoder
city_encoder = LabelEncoder()
X_train['city'] = city_encoder.fit_transform(X_train['city'])
X_test['city'] = city_encoder.transform(X_test['city'])
display(X_train.head())

size_encoder = LabelEncoder()
X_train['size'] = size_encoder.fit_transform(X_train['size'])
X_test['size'] = size_encoder.transform(X_test['size'])
display(X_train.head())

# checking for class imbalance
print(y_train.value_counts())

# training
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)

# print confusion matrix
print(confusion_matrix(y_test, y_pred))
# 2. Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 3. Plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(recall_score(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
