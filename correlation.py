import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)
    
    return df.loc[(df[col] > q1) & (df[col] < q3)]

vehicles = pd.read_csv("vehicles.csv")

df = pd.DataFrame(vehicles)

df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")

df["region"] = df["region"].where(df["region"].isin(df["region"].value_counts()[:50].index.tolist()), "other")
df["manufacturer"] = df["manufacturer"].where(df["manufacturer"].isin(df["manufacturer"].value_counts()[:20].index.tolist()), "other")

label_encoder = preprocessing.LabelEncoder()
df["region"] = label_encoder.fit_transform(df["region"])
df["manufacturer"] = label_encoder.fit_transform(df["manufacturer"])
df["cylinders"] = label_encoder.fit_transform(df["cylinders"])
df["state"] = label_encoder.fit_transform(df["state"])

corr = df.select_dtypes("number").corr()


sns.heatmap(corr)

plt.show()