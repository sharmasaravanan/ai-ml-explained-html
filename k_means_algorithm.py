import pandas as pd
from IPython.core.display_functions import display
from sklearn.cluster import KMeans

df = pd.read_csv("train.csv")
print("Displaying training data")
display(df.head())

# checking for null values
print(df.isnull().any().sum())
print(df.isna().any().sum())
print(df.shape)

# printing unique values of class
print(df['class'].unique() )

display(df.groupby('class').count())

features = df.drop('class', axis=1)
display(features.head())

model = KMeans(n_clusters=3)
model.fit(features)

labels = model.labels_
df['predicted'] = labels
display(df.head())
display(df.groupby('predicted').count())

df.to_csv("train_with_predicted.csv", index=False)