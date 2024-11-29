import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# function to remove outliers from data
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)

    return df.loc[(df[col] > q1) & (df[col] < q3)]

# opens the CSV file
vehicles = pd.read_csv("vehicles.csv")
df = pd.DataFrame(vehicles)

label_encoder = LabelEncoder()

# displays basic information about the dataframe
print(df.head())
print(df.shape)
print(df.info())

plt.figure()
df_corr = df
columns = list(df_corr.columns)
for column in columns:
    df_corr[column] = label_encoder.fit_transform(df_corr[column]) 

corr1 = df_corr.select_dtypes("number").corr()
sns.heatmap(corr1, annot=True)


# drops all columns that are irrelevant to a car's price
df = df.drop(columns=["id", "url", "region_url", "VIN", "image_url", "description", "posting_date"])

# displays the number of unique values for each remaining column
print(df.nunique())
df = df.drop(columns=["region", "county", "lat", "long"])

# removes outliers from the data as well as ensuring the entries all contain a sensible minimum price
df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")
df = df.loc[(df["price"] > 500)]

# converting the year column into an age column
df["age"] = 2024 - df["year"]
df = df.drop(columns=["year"])

# printing out the information of the dataframe after current preprocessing
print(df.info())

plt.figure()
df_corr = df
columns = list(df_corr.columns)
for column in columns:
    df_corr[column] = label_encoder.fit_transform(df_corr[column]) 

corr2 = df_corr.select_dtypes("number").corr()
sns.heatmap(corr2, annot=True)

# drop features that have a negligent effect on the price
df = df.drop(columns=["model", "paint_color"])

# find the average price of a car per state
price_per_state = df.groupby("state", as_index=False)["price"].mean()
print(price_per_state)

# encode the states to prepare them for clustering
price_per_state["encoded_state"] = label_encoder.fit_transform(price_per_state["state"])
clustering_data = price_per_state[["encoded_state", "price"]]

# split the data into three clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(clustering_data)
print(kmeans.labels_)

# adding the clustering labels to the clustering_data dataset
clustering_data["cluster"] = kmeans.labels_

print(clustering_data.info())
print(clustering_data.head())

# getting the data that must will be merged into the main dataframe
# completing the taks this way ensures better management of all the data
cluster_data_to_merge = clustering_data[["encoded_state", "cluster"]]
print(cluster_data_to_merge)

# get the state and its associated cluster label together
merging_data = pd.merge(cluster_data_to_merge, price_per_state, on="encoded_state", how="left")
merging_data = merging_data.drop(columns=["encoded_state", "price"])
print(merging_data)

# merge the clusters into the main dataframe and drop the state column
df = pd.merge(df, merging_data, on="state", how="left")
df = df.drop(columns=["state"])

print(df.info())

plt.figure()
df_corr = df
columns = list(df_corr.columns)
for column in columns:
    df_corr[column] = label_encoder.fit_transform(df_corr[column]) 

corr3 = df_corr.select_dtypes("number").corr()
sns.heatmap(corr3, annot=True)

df = pd.get_dummies(df)

X = df.drop(columns=["price"])
y = np.log1p(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regression = LinearRegression()
regression.fit(X_train, y_train)
score = regression.score(X_test, y_test)
print(score)

y_pred = np.expm1(regression.predict(X_test))
df_test = X_test.copy()
df_test["price"] = y_test
df_test["pred"] = y_pred

fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(10, 5))

ax1.scatter(df_test["age"], df_test["price"], color="blue")
ax2.scatter(df_test["odometer"], df_test["price"], color="green")
ax1.plot(df_test["age"], df_test["pred"], color="red")
ax2.plot(df_test["odometer"], df_test["pred"], color="red")

ax1.set_xlabel("Age")
ax1.set_ylabel("Price")
ax2.set_xlabel("Mileage")
ax2.set_ylabel("Price")

plt.show()