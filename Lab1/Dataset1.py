import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = fetch_california_housing(as_frame=True)
df = data.frame

print("Original Dataset:")
print(df.head())

df = df.fillna(df.median(numeric_only=True))

df = df.drop_duplicates()

for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

cols = df.select_dtypes(include=np.number).columns

minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[cols] = minmax.fit_transform(df[cols])

standard = StandardScaler()
df_standard = df.copy()
df_standard[cols] = standard.fit_transform(df[cols])

plt.figure(figsize=(8, 4))
df["HouseAge"].value_counts().head(10).plot(kind="bar")
plt.title("Bar Plot - House Age")
plt.xlabel("House Age")
plt.ylabel("Count")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df[cols], ax=axes[0])
axes[0].set_title("Before Normalization")

sns.boxplot(data=df_minmax[cols], ax=axes[1])
axes[1].set_title("After Min-Max Normalization")

sns.boxplot(data=df_standard[cols], ax=axes[2])
axes[2].set_title("After Z-score Standardization")

plt.tight_layout()
plt.show()

features = ["MedInc", "HouseAge", "AveRooms"]

for col in features:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"{col} - Original")

    sns.histplot(df_minmax[col], kde=True, ax=axes[1])
    axes[1].set_title(f"{col} - Min-Max")

    sns.histplot(df_standard[col], kde=True, ax=axes[2])
    axes[2].set_title(f"{col} - Z-score")

    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(
    x=df["MedInc"], y=df["MedHouseVal"], ax=axes[0]
)
axes[0].set_title("Original Data")

sns.scatterplot(
    x=df_minmax["MedInc"], y=df_minmax["MedHouseVal"], ax=axes[1]
)
axes[1].set_title("Min-Max Normalization")

sns.scatterplot(
    x=df_standard["MedInc"], y=df_standard["MedHouseVal"], ax=axes[2]
)
axes[2].set_title("Z-score Standardization")

plt.tight_layout()
plt.show()

print("\nMin-Max Normalized Data:")
print(df_minmax.head())

print("\nZ-score Standardized Data:")
print(df_standard.head())

print("\nPreprocessing Completed Successfully!")
