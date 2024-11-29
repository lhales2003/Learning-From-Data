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

df = df.loc[(df["price"] > 0)]
df = remove_outliers(df, "odometer")
df = remove_outliers(df, "price")

df["age"] = 2024 - df["year"]

df = df.drop(columns=["year"])

unique_counts = df.nunique()
print(unique_counts)
# df["region"] = df["region"].where(df["region"].isin(df["region"].value_counts()[:50].index.tolist()), "other")
# df["manufacturer"] = df["manufacturer"].where(df["manufacturer"].isin(df["manufacturer"].value_counts()[:20].index.tolist()), "other")

label_encoder = preprocessing.LabelEncoder()
# df["region"] = label_encoder.fit_transform(df["region"])
# df["manufacturer"] = label_encoder.fit_transform(df["manufacturer"])
# df["model"] = label_encoder.fit_transform(df["model"])
# df["fuel"] = label_encoder.fit_transform(df["fuel"])
# df["title_status"] = label_encoder.fit_transform(df["title_status"])
# df["type"] = label_encoder.fit_transform(df["type"])
# df["drive"] = label_encoder.fit_transform(df["drive"])
# df["condition"] = label_encoder.fit_transform(df["condition"])
# df["cylinders"] = label_encoder.fit_transform(df["cylinders"])
# df["state"] = label_encoder.fit_transform(df["state"])

columns = ["url", "region", "region_url", "manufacturer", "model", "condition", "cylinders", "fuel", "title_status", "transmission", "VIN", "drive", "size", "type", "paint_color", "image_url", "description", "county", "state", "posting_date"]

for column in columns:
    df[column] = label_encoder.fit_transform(df[column])  

corr = df.select_dtypes("number").corr()


sns.heatmap(corr, annot=True)

plt.show()