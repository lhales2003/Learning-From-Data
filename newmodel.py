import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    q1 = q1 - (1.5 * iqr)
    q3 = q3 + (1.5 * iqr)
    
    return df.loc[(df[col] > q1) & (df[col] < q3)]

vehicles = pd.read_csv("vehicles.csv")
df = pd.DataFrame(vehicles)

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

df.drop(columns=["county"])

no_of_regions = df["region"].value_counts()
print(no_of_regions)