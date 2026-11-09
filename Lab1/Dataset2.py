import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = fetch_openml(data_id=42165, as_frame=True).frame

print("Original Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sum())

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

df[cat_cols] = SimpleImputer(
    strategy="most_frequent"
).fit_transform(df[cat_cols])

df = df.drop_duplicates()

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

minmax = MinMaxScaler()
df_minmax = df.copy()

df_minmax[num_cols] = minmax.fit_transform(df[num_cols])

standard = StandardScaler()
df_standard = df.copy()

df_standard[num_cols] = standard.fit_transform(df[num_cols])

plt.figure(figsize=(8, 4))
df["OverallQual"].value_counts().sort_index().plot(kind="bar")

plt.title("House Quality Distribution")
plt.xlabel("Overall Quality")
plt.ylabel("Count")
plt.show()

cols = ["LotArea", "GrLivArea", "SalePrice"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.boxplot(data=df[cols], ax=axes[0])
axes[0].set_title("Before Normalization")

sns.boxplot(data=df_minmax[cols], ax=axes[1])
axes[1].set_title("After Min-Max Normalization")

sns.boxplot(data=df_standard[cols], ax=axes[2])
axes[2].set_title("After Z-score Standardization")

plt.tight_layout()
plt.show()

for col in cols:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title("Original " + col)

    sns.histplot(df_minmax[col], kde=True, ax=axes[1])
    axes[1].set_title("Min-Max " + col)

    sns.histplot(df_standard[col], kde=True, ax=axes[2])
    axes[2].set_title("Z-score " + col)

    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.scatterplot(
    x=df["GrLivArea"], y=df["SalePrice"], ax=axes[0]
)
axes[0].set_title("Original Data")

sns.scatterplot(
    x=df_minmax["GrLivArea"], y=df_minmax["SalePrice"], ax=axes[1]
)
axes[1].set_title("Min-Max Normalization")

sns.scatterplot(
    x=df_standard["GrLivArea"], y=df_standard["SalePrice"], ax=axes[2]
)
axes[2].set_title("Z-score Standardization")

plt.tight_layout()
plt.show()

print("\nPreprocessed Dataset:")
print(df.head())

print("\nMin-Max Normalized Dataset:")
print(df_minmax.head())

print("\nZ-score Standardized Dataset:")
print(df_standard.head())

print("\nRemaining Missing Values:", df.isnull().sum().sum())
print("Remaining Duplicates:", df.duplicated().sum())

print("\nPreprocessing Completed Successfully!")
