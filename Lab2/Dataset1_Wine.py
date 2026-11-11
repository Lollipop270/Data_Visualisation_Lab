import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

wine = load_wine(as_frame=True)
df = wine.frame.copy()

df["class"] = df["target"].map(
    dict(enumerate(wine.target_names))
)

df.drop(columns=["target"], inplace=True)

features = wine.feature_names

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nCLASS DISTRIBUTION")
print(df["class"].value_counts())

df[features] = df[features].fillna(
    df[features].median()
)

Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

outliers = (
    (df[features] < Q1 - 1.5 * IQR) |
    (df[features] > Q3 + 1.5 * IQR)
)

print("\nOUTLIER COUNT PER FEATURE")
print(outliers.sum().sort_values(ascending=False))

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="class", hue="class",
              palette="viridis", legend=False)

plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

df["alcohol_group"] = pd.cut(
    df["alcohol"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

stacked = pd.crosstab(
    df["alcohol_group"], df["class"]
)

stacked.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5),
    colormap="viridis"
)

plt.title("Wine Classes by Alcohol Level")
plt.xlabel("Alcohol Group")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

selected = [
    "alcohol", "malic_acid", "ash",
    "magnesium", "total_phenols",
    "flavanoids", "color_intensity"
]

scaler = StandardScaler()

scaled = pd.DataFrame(
    scaler.fit_transform(df[features]),
    columns=features
)

scaled["class"] = df["class"].values

line_data = scaled.groupby("class")[selected].mean().T

line_data.plot(
    figsize=(11, 5),
    marker="o"
)

plt.title("Average Standardized Chemical Properties by Wine Class")
plt.xlabel("Chemical Features")
plt.ylabel("Standardized Mean")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="flavanoids",
    hue="class",
    palette="viridis",
    s=80
)

plt.title("Alcohol vs Flavanoids")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))

plt.scatter(
    df["alcohol"],
    df["flavanoids"],
    s=df["proline"] / 10,
    c=df["target"] if "target" in df.columns else
      pd.Categorical(df["class"]).codes,
    cmap="viridis",
    alpha=0.6
)

plt.xlabel("Alcohol")
plt.ylabel("Flavanoids")
plt.title("Bubble Chart: Alcohol, Flavanoids and Proline")
plt.colorbar(label="Wine Class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="alcohol",
    hue="class",
    kde=True,
    bins=20,
    element="step"
)

plt.title("Distribution of Alcohol Content")
plt.tight_layout()
plt.show()

box_features = [
    "alcohol", "malic_acid",
    "magnesium", "flavanoids",
    "color_intensity", "proline"
]

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df[box_features],
    orient="h",
    palette="Set2"
)

plt.title("Box Plot of Wine Chemical Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(13, 10))

sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Wine Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

radar_features = [
    "alcohol", "malic_acid",
    "magnesium", "flavanoids",
    "color_intensity", "proline"
]

radar_df = pd.DataFrame(
    scaler.fit_transform(df[radar_features]),
    columns=radar_features
)

radar_df["class"] = df["class"].values

radar_means = radar_df.groupby("class")[radar_features].mean()

angles = np.linspace(
    0, 2 * np.pi, len(radar_features), endpoint=False
).tolist()

angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw={"polar": True}
)

for name, values in radar_means.iterrows():
    vals = values.tolist()
    vals += vals[:1]

    ax.plot(angles, vals, linewidth=2, label=name)
    ax.fill(angles, vals, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_features)
ax.set_title("Radar Chart: Wine Class Profiles", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.show()

df["class"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(7, 7),
    cmap="viridis"
)

plt.title("Percentage Distribution of Wine Classes")
plt.ylabel("")
plt.show()

treemap_df = df.groupby(
    ["class", "alcohol_group"],
    observed=True
).size().reset_index(name="count")

fig = px.treemap(
    treemap_df,
    path=["class", "alcohol_group"],
    values="count",
    color="count",
    color_continuous_scale="Viridis",
    title="Treemap of Wine Classes and Alcohol Groups"
)

fig.show()

word_text = """
Wine Alcohol Malic Acid Ash Magnesium
Total Phenols Flavanoids Nonflavanoid Phenols
Proanthocyanins Color Intensity Hue
OD280 Proline Wine Class Chemical Composition
Wine Quality Phenolic Compounds
"""

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap="viridis"
).generate(word_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Wine Dataset Word Cloud")
plt.show()

print("\nCLASS-WISE FEATURE MEANS")
print(df.groupby("class")[features].mean().round(2))

print("\nTOP POSITIVE CORRELATIONS")

corr = df[features].corr()
corr_pairs = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
).stack()

print(corr_pairs.sort_values(ascending=False).head(10))

print("\nTOP NEGATIVE CORRELATIONS")
print(corr_pairs.sort_values().head(10))

print("\nSKEWNESS")
print(df[features].skew().sort_values(ascending=False))

print("\nMOST FREQUENT OUTLIER FEATURES")
print(outliers.sum().sort_values(ascending=False).head(5))

print("\nWINE DATASET EDA COMPLETED")
