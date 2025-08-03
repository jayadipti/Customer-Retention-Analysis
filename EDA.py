import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# Load your dataset
df = pd.read_csv("healthcare_churn.csv")

# Replace inf/-inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns with more than 50% missing values
threshold = len(df) * 0.5
df = df.loc[:, df.isnull().sum() < threshold]

# Fill missing numeric data with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables for correlation analysis
df_encoded = df.copy()
df_encoded["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df_encoded["PlanType"] = df["PlanType"].map({"Basic": 0, "Standard": 1, "Premium": 2})
df_encoded["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# --- Plot 1: Clustered Correlation Heatmap ---
df_numeric = df_encoded.select_dtypes(include=[np.number])
try:
    if not df_numeric.empty and df_numeric.shape[1] > 1:
        cg = sns.clustermap(df_numeric.corr(), linewidths=0.5, figsize=(14, 10), cmap="coolwarm", annot=True)
        plt.suptitle("Clustered Correlation Heatmap", y=1.02, fontsize=16)
        plt.show()
    else:
        print("Not enough numeric data for correlation heatmap.")
except Exception as e:
    print("Clustered heatmap skipped due to error:", e)

# --- Plot 2: Interactive Correlation Heatmap using Plotly ---
try:
    corr = df_numeric.corr()
    if not corr.empty:
        fig = ff.create_annotated_heatmap(
            z=corr.values.round(2),
            x=list(corr.columns),
            y=list(corr.columns),
            annotation_text=corr.round(2).astype(str).values,
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(title_text="Interactive Correlation Heatmap", height=800)
        fig.show()
    else:
        print("No numeric correlation data to plot.")
except Exception as e:
    print("Plotly correlation heatmap error:", e)

# --- Plot 3: Churn Distribution Plot ---
if "Churn" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", palette="pastel")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
else:
    print("Churn column not found in data.")

# --- Plot 4: Usage vs Churn KDE Plot ---
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="Usage", hue="Churn", fill=True, palette="Set2")
plt.title("Usage Distribution by Churn")
plt.xlabel("Usage")
plt.grid(True)
plt.tight_layout()
plt.show()
