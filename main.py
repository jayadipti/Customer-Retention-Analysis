import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path="healthcare_churn.csv"):
    """Load and clean the dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def encode_categorical_data(df):
    """Encode categorical values"""
    df_encoded = df.copy()
    df_encoded["Gender"] = df_encoded["Gender"].map({"Male": 0, "Female": 1})
    df_encoded["PlanType"] = df_encoded["PlanType"].map({"Basic": 0, "Standard": 1, "Premium": 2})
    df_encoded["Churn"] = df_encoded["Churn"].map({"No": 0, "Yes": 1})
    return df_encoded

def analyze_churn_rate(df):
    """Print churn rate"""
    print("=" * 60)
    print("CUSTOMER RETENTION ANALYSIS")
    print("=" * 60)

    churn_rate = df["Churn"].value_counts(normalize=True) * 100
    print(f"\nOverall Churn Rate:")
    print(f"   • Churned Customers: {churn_rate[1]:.1f}% ({df['Churn'].value_counts()[1]} customers)")
    print(f"   • Retained Customers: {churn_rate[0]:.1f}% ({df['Churn'].value_counts()[0]} customers)")
    return churn_rate

def analyze_metrics_by_churn(df):
    """Print key metrics by churn"""
    print("\n" + "=" * 60)
    print("AVERAGE METRICS BY CHURN STATUS")
    print("=" * 60)

    metrics = ["Age", "Usage", "Tenure", "ServiceLag", "Complaints"]
    churn_analysis = df.groupby("Churn")[metrics].mean()
    print("\nMean values by churn status:")
    print(churn_analysis.round(2))
    return churn_analysis

def create_visualizations(df, df_encoded):
    """Create visual plots with perfect layout"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    plt.style.use('seaborn-v0_8')

    # Using constrained_layout for zero overlap
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle('Customer Retention Analysis Dashboard', fontsize=18, fontweight='bold')

    # 1. Plan Type vs Churn Rate
    plan_churn = df_encoded.groupby('PlanType')['Churn'].mean() * 100
    axes[0, 0].bar(plan_churn.index, plan_churn.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Churn Rate by Plan Type', fontweight='bold')
    axes[0, 0].set_xlabel('Plan Type')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].set_xticklabels(['Basic', 'Standard', 'Premium'])
    axes[0, 0].grid(True, alpha=0.3)

    for i, v in enumerate(plan_churn.values):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    # 2. Service Lag vs Churn
    sns.boxplot(x="Churn", y="ServiceLag", data=df_encoded, ax=axes[0, 1], palette=['#FF6B6B', '#4ECDC4'])
    axes[0, 1].set_title('Service Lag vs Churn', fontweight='bold')
    axes[0, 1].set_xlabel('Churn Status')
    axes[0, 1].set_ylabel('Service Lag (days)')
    axes[0, 1].set_xticklabels(['Retained', 'Churned'])

    # 3. Usage vs Churn
    sns.boxplot(x="Churn", y="Usage", data=df_encoded, ax=axes[1, 0], palette=['#FF6B6B', '#4ECDC4'])
    axes[1, 0].set_title('Usage vs Churn', fontweight='bold')
    axes[1, 0].set_xlabel('Churn Status')
    axes[1, 0].set_ylabel('Usage')
    axes[1, 0].set_xticklabels(['Retained', 'Churned'])

    # 4. Age Distribution
    sns.histplot(data=df_encoded, x="Age", hue="Churn", bins=20, ax=axes[1, 1],
                 palette=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    axes[1, 1].set_title('Age Distribution by Churn', fontweight='bold')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend(['Retained', 'Churned'])

    plt.show()

    # Additional summaries
    complaint_crosstab = pd.crosstab(df["Complaints"], df["Churn"], normalize='index') * 100
    print("\nChurn Rate by Number of Complaints:")
    print(complaint_crosstab.round(1))

    tenure_bins = pd.cut(df['Tenure'], bins=[0, 12, 24, 36, 48, 60],
                         labels=['0-12m', '13-24m', '25-36m', '37-48m', '49-60m'])
    tenure_churn = df_encoded.groupby(tenure_bins)['Churn'].mean() * 100
    print("\nChurn Rate by Tenure:")
    print(tenure_churn.round(1))

def generate_insights(df, df_encoded, churn_rate, churn_analysis):
    """Print actionable business insights"""
    print("\n" + "=" * 60)
    print("ACTIONABLE INSIGHTS FOR RETENTION STRATEGY")
    print("=" * 60)

    insights = []

    if churn_rate[1] > 25:
        insights.append("High churn rate (>25%) detected! Retention intervention needed.")
    elif churn_rate[1] > 15:
        insights.append("Moderate churn detected. Review service and engagement strategies.")
    else:
        insights.append("Healthy churn rate. Continue monitoring.")

    service_lag_diff = churn_analysis.loc[1, 'ServiceLag'] - churn_analysis.loc[0, 'ServiceLag']
    if service_lag_diff > 1:
        insights.append(f"Churned users face longer service delays (+{service_lag_diff:.1f} days). Improve response times.")

    usage_diff = churn_analysis.loc[0, 'Usage'] - churn_analysis.loc[1, 'Usage']
    if usage_diff > 20:
        insights.append(f"Lower usage detected in churned users (-{usage_diff:.0f} units). Promote platform engagement.")

    complaints_diff = churn_analysis.loc[1, 'Complaints'] - churn_analysis.loc[0, 'Complaints']
    if complaints_diff > 0.2:
        insights.append(f"Churned customers report more complaints (+{complaints_diff:.1f}). Strengthen support.")

    plan_churn_rates = df_encoded.groupby('PlanType')['Churn'].mean()
    if plan_churn_rates[0] > plan_churn_rates[2]:
        insights.append("Basic plan users churn more than Premium. Upsell or incentivize upgrades.")

    age_churn = df_encoded.groupby('Churn')['Age'].mean()
    if age_churn[1] > age_churn[0] + 5:
        insights.append("Older customers churn more. Tailor support for senior users.")

    for i, msg in enumerate(insights, 1):
        print(f"{i}. {msg}")
    return insights

def main():
    """Run everything end to end"""
    print("Starting Customer Retention Analysis...\n")
    df = load_and_prepare_data()
    if df is None:
        return
    df_encoded = encode_categorical_data(df)
    churn_rate = analyze_churn_rate(df)
    churn_analysis = analyze_metrics_by_churn(df_encoded)
    create_visualizations(df, df_encoded)
    insights = generate_insights(df, df_encoded, churn_rate, churn_analysis)
    print("\nAnalysis Completed | Total Customers:", len(df))
    print("Total Key Insights Generated:", len(insights))
    print("\nNext steps: Implement the suggested retention strategies!")

if __name__ == "__main__":
    main()
