import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import os
import re
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def sanitize_filename(tag):
    """Convert tag names to valid filenames"""
    return re.sub(r'[\\/*?:"<>|()]', '_', tag)

def create_dirs():
    """Create necessary directories for saving results"""
    dirs = {
        'distributions': 'eda_results/distributions',
        'correlation': 'eda_results/correlation',
        'categorical': 'eda_results/categorical',
        'summary': 'eda_results/summary'
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def load_data(file_path):
    """Load and prepare the dataset with comprehensive data cleaning"""
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Data quality check
        print(f"Data quality check:")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        print(f"   • Duplicate rows: {df.duplicated().sum()}")
        
        # Replace inf/-inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop columns with more than 50% missing values
        threshold = len(df) * 0.5
        df = df.loc[:, df.isnull().sum() < threshold]
        
        # Fill missing numeric data with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"   • Filled missing values in {col} with median")
        
        # Fill missing categorical data with mode
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"   • Filled missing values in {col} with mode")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_distributions(df, dirs):
    """Create and save distribution plots for all numeric columns"""
    print("\nCreating distribution plots...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    saved_plots = []
    
    for col in numeric_cols:
        try:
            data = df[col].dropna()
            if data.empty:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Create subplot with histogram and boxplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with KDE
            sns.histplot(data, bins=30, kde=True, color='skyblue', alpha=0.7, ax=ax1)
            ax1.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax1.set_xlabel(col, fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Boxplot
            sns.boxplot(y=data, color='lightcoral', ax=ax2)
            ax2.set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
            ax2.set_ylabel(col, fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nMin: {data.min():.2f}\nMax: {data.max():.2f}'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            safe_name = sanitize_filename(col)
            plt.savefig(f"{dirs['distributions']}/dist_{safe_name}.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            saved_plots.append(col)
            print(f"   Saved distribution plot for {col}")
            
        except Exception as e:
            print(f"   Failed to create distribution plot for {col}: {str(e)}")
    
    print(f"Successfully created {len(saved_plots)} distribution plots")
    return saved_plots

def save_correlation(df, dirs):
    """Create and save correlation analysis"""
    print("\nCreating correlation analysis...")
    
    # Create encoded version for correlation
    df_encoded = df.copy()
    
    # Encode categorical variables
    if 'Gender' in df_encoded.columns:
        df_encoded["Gender"] = df_encoded["Gender"].map({"Male": 0, "Female": 1})
    if 'PlanType' in df_encoded.columns:
        df_encoded["PlanType"] = df_encoded["PlanType"].map({"Basic": 0, "Standard": 1, "Premium": 2})
    if 'Churn' in df_encoded.columns:
        df_encoded["Churn"] = df_encoded["Churn"].map({"No": 0, "Yes": 1})

    numeric_df = df_encoded.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("   Not enough numeric columns for correlation matrix.")
        return
    
    try:
        # Clustered correlation heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{dirs['correlation']}/clustered_corr.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("   Saved clustered correlation heatmap")
        
    except Exception as e:
        print(f"   Clustered heatmap failed: {str(e)}")

    try:
        # Interactive correlation heatmap
        corr = numeric_df.corr().round(3)
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.columns),
            annotation_text=corr.astype(str).values,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False
        )
        fig.update_layout(
            title_text="Interactive Correlation Matrix",
            title_x=0.5,
            height=800,
            width=800
        )
        fig.write_html(f"{dirs['correlation']}/interactive_corr.html")
        print("   Saved interactive correlation heatmap")
        
    except Exception as e:
        print(f"   Interactive heatmap failed: {str(e)}")

def save_categorical_plots(df, dirs):
    """Create and save categorical analysis plots"""
    print("\nCreating categorical analysis plots...")
    
    # Churn distribution
    if 'Churn' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            churn_counts = df['Churn'].value_counts()
            colors = ['#4ECDC4', '#FF6B6B']
            
            plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Churn Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.savefig(f"{dirs['categorical']}/churn_distribution.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
            print("   Saved churn distribution plot")
            
        except Exception as e:
            print(f"   Churn distribution failed: {str(e)}")

    # Usage vs Churn KDE
    if 'Usage' in df.columns and 'Churn' in df.columns:
        try:
            plt.figure(figsize=(12, 8))
            sns.kdeplot(data=df, x="Usage", hue="Churn", fill=True, 
                       palette=['#4ECDC4', '#FF6B6B'], alpha=0.7)
            plt.title('Usage Distribution by Churn Status', fontsize=16, fontweight='bold')
            plt.xlabel('Usage', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(['Retained', 'Churned'])
            plt.savefig(f"{dirs['categorical']}/usage_vs_churn.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
            print("   Saved usage vs churn plot")
            
        except Exception as e:
            print(f"   Usage vs Churn KDE failed: {str(e)}")

    # Plan Type analysis
    if 'PlanType' in df.columns and 'Churn' in df.columns:
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplot with plan type distribution and churn rate
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plan type distribution
            plan_counts = df['PlanType'].value_counts()
            ax1.bar(plan_counts.index, plan_counts.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('Plan Type Distribution', fontweight='bold')
            ax1.set_xlabel('Plan Type')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            
            # Churn rate by plan type
            plan_churn = df.groupby('PlanType')['Churn'].mean() * 100
            ax2.bar(plan_churn.index, plan_churn.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax2.set_title('Churn Rate by Plan Type', fontweight='bold')
            ax2.set_xlabel('Plan Type')
            ax2.set_ylabel('Churn Rate (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(plan_churn.values):
                ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{dirs['categorical']}/plan_type_analysis.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
            print("   Saved plan type analysis")
            
        except Exception as e:
            print(f"   Plan type analysis failed: {str(e)}")

def create_summary_report(df, dirs):
    """Create a comprehensive summary report"""
    print("\nCreating summary report...")
    
    try:
        # Basic statistics
        summary_stats = df.describe()
        
        # Save summary statistics
        with open(f"{dirs['summary']}/summary_report.txt", 'w') as f:
            f.write("CUSTOMER RETENTION ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total customers: {len(df)}\n")
            f.write(f"Total features: {len(df.columns)}\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
            
            f.write("COLUMN INFORMATION:\n")
            for col in df.columns:
                f.write(f"{col}: {df[col].dtype}\n")
            f.write("\n")
            
            f.write("NUMERIC FEATURES SUMMARY:\n")
            f.write(summary_stats.to_string())
            f.write("\n\n")
            
            f.write("CATEGORICAL FEATURES SUMMARY:\n")
            for col in df.select_dtypes(include=['object']).columns:
                f.write(f"{col}:\n")
                f.write(df[col].value_counts().to_string())
                f.write("\n\n")
            
            if 'Churn' in df.columns:
                churn_rate = df['Churn'].value_counts(normalize=True) * 100
                f.write("CHURN ANALYSIS:\n")
                f.write(f"Churn rate: {churn_rate['Yes']:.1f}%\n")
                f.write(f"Retention rate: {churn_rate['No']:.1f}%\n")
        
        print("   Saved summary report")
        
    except Exception as e:
        print(f"   Summary report failed: {str(e)}")

def main():
    """Main execution function"""
    print("Starting Comprehensive EDA...")
    print("=" * 50)
    
    # Create directories
    dirs = create_dirs()
    print("Created output directories")
    
    # Load data
    df = load_data("healthcare_churn.csv")
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Perform analysis
    saved_distributions = save_distributions(df, dirs)
    save_correlation(df, dirs)
    save_categorical_plots(df, dirs)
    create_summary_report(df, dirs)
    
    print("\n" + "=" * 50)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Total customers analyzed: {len(df)}")
    print(f"Distribution plots created: {len(saved_distributions)}")
    print(f"Results saved in 'eda_results' folder")
    print("\nGenerated files:")
    print("   • Distribution plots: eda_results/distributions/")
    print("   • Correlation analysis: eda_results/correlation/")
    print("   • Categorical analysis: eda_results/categorical/")
    print("   • Summary report: eda_results/summary/")
    print("\nNext: Run main.py for detailed retention analysis!")

if __name__ == "__main__":
    main()
