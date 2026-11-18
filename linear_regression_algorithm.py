from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display

# loading dataset
url = "https://raw.githubusercontent.com/sharmasaravanan/Machine_learning/refs/heads/master/Advertising.csv"
dataset = pd.read_csv(url)
print("Top-5 rows of dataset")
display(dataset.head())

# checking for null values
print(dataset.isnull().sum())
print(dataset.isna().sum())
print("Five number properties of dataset")
display(dataset.describe())

# checking for outliers
# dataset.plot(kind="box")
# plt.show()
X = dataset[['TV',"radio", "newspaper"]]
y = dataset['sales']

print("Shape of dataset", dataset.shape)
# display(X.head())

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)

# training
model = LinearRegression()
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
# RMSE
RMSE = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# r2_score
score = r2_score(y_test, y_pred)
print("score:", score)

# equation
# y = m1*x1 + m2*x2 + m3*x3 + c
print(model.coef_)
print(model.intercept_)

# let's find out how much we will get if we invest 100 in each
testSet = [[100,100,100]]
sale_value = model.predict(testSet)
print(sale_value)