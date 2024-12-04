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

def main():
    # opens the CSV file
    vehicles = pd.read_csv("vehicles.csv")
    df = pd.DataFrame(vehicles)

    # displays basic information about the dataframe
    print(df.head())
    print(df.shape)
    print(df.info())

    label_encoder = LabelEncoder()

    # drops all columns that are irrelevant to a car's price
    df = df.drop(columns=["id", "url", "region_url", "VIN", "image_url", "description", "posting_date"])
    print("After dropping columns:\n", df.isnull().sum())

    # displays the number of unique values for each remaining column
    print(df.nunique())
    df = df.drop(columns=["region", "county", "lat", "long"])
    print("After dropping columns:\n", df.isnull().sum())

    # removes outliers from the data as well as ensuring the entries all contain a sensible minimum price
    df = remove_outliers(df, "odometer")
    df = remove_outliers(df, "price")
    df = df.loc[(df["price"] > 500)]
    print(df.isnull().sum())

    # converting the year column into an age column
    df["age"] = 2024 - df["year"]
    df = df.drop(columns=["year"])
    print("After changing year:\n", df.isnull().sum())

    # printing out the information of the dataframe after current preprocessing
    print(df.info())

    # drop features that have a negligent effect on the price
    df = df.drop(columns=["model", "fuel", "paint_color"])

    print("Before removing cols:\n", df.info())
    df = df.dropna(subset=["age"])
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna("Unknown")
    print("After removing cols:\n", df.info())

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

    df = pd.get_dummies(df)

    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test