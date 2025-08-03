Customer Retention Analysis

This project performs an in-depth exploratory data analysis (EDA) on a healthcare customer dataset to uncover key drivers of churn and support the design of effective retention strategies.

 Project Objective

- Identify patterns in customer behavior that correlate with churn
- Analyze service usage, complaints, and tenure as indicators of drop-off
- Provide actionable insights for improving customer retention

 Dataset Overview

The dataset contains anonymized healthcare customer records with the following columns:

| Column       | Description                             |
|--------------|-----------------------------------------|
| CustomerID   | Unique customer identifier              |
| Age          | Age of the customer                     |
| Gender       | Gender (Male/Female)                    |
| Tenure       | Number of months as a customer          |
| Usage        | Service usage (numeric units)           |
| PlanType     | Type of healthcare plan (Basic/Standard/Premium) |
| Complaints   | Number of complaints raised             |
| ServiceLag   | Average service delay in days           |
| Churn        | Whether the customer churned (Yes/No)   |


Technologies Used

- Python 3.10+
- Pandas for data wrangling
- Seaborn & Matplotlib for visualizations
- NumPy for numerical computations

 Data Preprocessing

- Handled missing values (critical vs. imputation strategy)
- Imputed numeric columns with median
- Imputed categorical columns with mode
- Removed whitespace from identifiers
- Applied value clipping to ensure data consistency

Exploratory Data Analysis

Key analyses included:

- Churn Rate Calculation  
- Usage vs. Churn (Boxplot)  
- Service Lag vs. Churn (Boxplot)  
- Churn by Plan Type (Bar chart)  
- Age Distribution by Churn Status  
- Churn Rate by Tenure Bucket and Complaints


 Insights Generated

- Customers with longer service lag are more likely to churn
- Lower usage levels correlate with higher churn probability
- **Premium plan users show lower churn rates than Basic
- **Senior customers tend to churn more
- Customers with more complaints have higher churn rates

These insights can guide targeted retention strategies such as:
- Improving service responsiveness
- Proactively engaging low-usage users
- Offering loyalty benefits for senior or long-tenure customers


# Run the analysis
python main.py
