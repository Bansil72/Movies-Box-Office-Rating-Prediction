# Complete Exploratory Data Analysis (EDA) and Predictive Modeling on Movies Dataset

This notebook demonstrates a complete end-to-end data analysis, starting from theoretical foundations and applying them through practical implementation.

---

## 1. Load Data

**Theory:** EDA is an approach to analyzing datasets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.

**Practical Implementation:** We will load our dataset, understand its shape, columns, find missing values, and observe basic descriptive statistics.

### Code: Import Libraries and Load Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = pd.read_csv('movies_dataset.csv')
print(f"Dataset Shape: {df.shape}")
df.head()
```

### Code: Basic Information

```python
# Basic Information
df.info()
```

### Code: Descriptive Statistics

```python
# Descriptive Statistics
df.describe()
```

### Code: Missing Values

```python
# Missing Values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]
```

---

## 2. Exploratory Data Analysis (EDA)

**Theory:** Visualizations help in finding patterns, detecting outliers, and understanding the distribution of data. 

**Practical Implementation:** We will visualize the distribution of budget and revenue, and explore categorical variables like 'Genre' and 'LeadActor', as well as understanding the correlation between features like 'NumVotesIMDb' and 'Global_BoxOfficeUSD'.

### 2.1 Univariate Analysis - Histograms, Count Plots

Examining individual variables to understand their distributions.

#### Code: Distribution of Budget and Global Box Office

```python
# Distribution of Budget and Global Box Office
from matplotlib.ticker import FuncFormatter

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Helper function to format y-axis counts entirely in Thousands
def thousands_formatter(x, pos):
    return f'{x / 1000:g}'

# Convert to Millions for better plot readability
budget_m = df['BudgetUSD'].dropna() / 1e6
box_office_m = df['Global_BoxOfficeUSD'].dropna() / 1e6

sns.histplot(budget_m, bins=50, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Budget (Millions USD)')
axes[0].set_xlabel('Budget (Millions USD)')
axes[0].set_ylabel('Count (Thousands)')

sns.histplot(box_office_m, bins=50, kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribution of Global Box Office (Millions USD)')
axes[1].set_xlabel('Global Box Office (Millions USD)')
axes[1].set_ylabel('Count (Thousands)')

# Use log scale on y-axis to see the distribution of both small and extremely large values
axes[0].set_yscale('log')
axes[1].set_yscale('log')

# IMPORTANT: Set formatter AFTER setting the scale, otherwise log scale overrides it
axes[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
axes[1].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

# Explicitly avoiding text annotations on bars to prevent overlapping and clutter
plt.tight_layout()
plt.show()
```

#### Code: Count of Movies by Genre

```python
# Univariate Analysis: Count of Movies by Genre
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Genre', order=df['Genre'].value_counts().index, palette='pastel')
plt.title('Number of Movies per Genre')
plt.xlabel('Count')
plt.ylabel('Genre')
ax = plt.gca()
ax.bar_label(ax.containers[0], fmt="%d", padding=3)
plt.show()
```

### 2.2 Bivariate Analysis - Bar Plots, Scatter Plots, Box Plots

Examining relationships between two variables.

#### Code: Top 10 Lead Actors by Average Global Box Office

```python
# Top 10 Lead Actors by Average Global Box Office
# We include 'LeadActor' to see if certain actors are associated with higher revenue as requested
# Converting to Millions USD for better visibility on the x-axis
top_actors_m = (df.groupby('LeadActor')['Global_BoxOfficeUSD'].mean() / 1e6).sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_actors_m.values, y=top_actors_m.index, palette='viridis')
plt.title('Top 10 Lead Actors by Average Global Box Office (Millions USD)')
plt.xlabel('Average Global Box Office (Millions USD)')
plt.ylabel('Lead Actor')

ax = plt.gca()
for i, v in enumerate(top_actors_m.values):
    # Format as e.g. "$1,500M" or "$250.5M"
    ax.text(v, i, f" ${v:,.1f}M", va="center", fontsize=10)

plt.show()
```

#### Code: Relationship between Number of Votes (IMDb) and Box Office

```python
# Relationship between Number of Votes (IMDb) and Box Office
# We include 'NumVotesIMDb' as a measure of popularity corresponding to the 'numofvoting' requirement
plt.figure(figsize=(10, 6))

df_scatter = df.copy()
df_scatter['BoxOffice_Millions'] = df_scatter['Global_BoxOfficeUSD'] / 1e6
df_scatter['Votes_Millions'] = df_scatter['NumVotesIMDb'] / 1e6

sns.scatterplot(data=df_scatter, x='Votes_Millions', y='BoxOffice_Millions', alpha=0.5)
plt.title('IMDb Votes vs Global Box Office (Millions)')
plt.xlabel('Number of IMDb Votes (Millions)')
plt.ylabel('Global Box Office (Millions USD)')
```

#### Code: Global Box Office Distribution Across Genres

```python
# Bivariate Analysis: Global Box Office distribution across Genres
plt.figure(figsize=(14, 6))

# Convert Box Office to Millions USD for readability
df_box = df.copy()
df_box['BoxOffice_Millions'] = df_box['Global_BoxOfficeUSD'] / 1e6

sns.boxplot(data=df_box, x='Genre', y='BoxOffice_Millions', palette='Set2')
plt.title('Global Box Office per Genre (Millions USD)')
plt.xlabel('Genre')
plt.ylabel('Global Box Office (Millions USD)')
plt.xticks(rotation=45)

ax = plt.gca()
medians = df_box.groupby(['Genre'])['BoxOffice_Millions'].median()
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    genre_name = text.get_text()
    if genre_name in medians:
        # Format the text inside the boxplot (median)
        ax.text(tick, medians[genre_name], f"${medians[genre_name]:,.1f}M", 
                horizontalalignment='center', size='small', color='w', weight='semibold')
plt.show()
```

### 2.3 Multivariate Analysis - Correlation Matrix (Heatmap), Scatter Plots with Hue

Analyzing relationships between more than two variables.

#### Code: Correlation Matrix

```python
# Correlation Matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.show()
```

#### Code: Budget vs Global Box Office colored by Genre

```python
# Multivariate Analysis: Budget vs Global Box Office colored by Genre
# Using a sample to speed up plotting for the large dataset
plt.figure(figsize=(14, 8))

# Convert values to Millions USD
sample_df = df.sample(min(10000, len(df)), random_state=42).copy()
sample_df['Budget_Millions'] = sample_df['BudgetUSD'] / 1e6
sample_df['BoxOffice_Millions'] = sample_df['Global_BoxOfficeUSD'] / 1e6

sns.scatterplot(data=sample_df, x='Budget_Millions', y='BoxOffice_Millions', hue='Genre', alpha=0.6, palette='Set1')
plt.title('Budget vs Global Box Office by Genre (Millions USD)')
plt.xlabel('Budget (Millions USD)')
plt.ylabel('Global Box Office (Millions USD)')
from matplotlib.ticker import FuncFormatter
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:g}M'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:g}M'))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
```

---

## 3. Data Cleaning

Before jumping into outlier detection and regression models, it's essential to clean the data by handling missing values and duplicate rows.

### Code: Clean Data

```python
# Data Cleaning
print('Initial dataset shape:', df.shape)

# 1. Drop Duplicates
df = df.drop_duplicates()
print('Shape after dropping duplicates:', df.shape)

# 2. Handle Missing Values
# Imputing numerical columns with median and categorical columns with mode
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print('Missing values after cleaning:', df.isnull().sum().sum())
```

---

## 4. Detect Outliers using Boxplots

Identifying outliers in the numerical features which might affect the machine learning models.

### Code: Visualize Outliers

```python
# Detect Outliers using Boxplots
plt.figure(figsize=(16, 12))
features_to_check = ['BudgetUSD', 'Global_BoxOfficeUSD', 'Opening_Day_SalesUSD', 'One_Week_SalesUSD', 'NumVotesIMDb', 'NumVotesRT']

for i, feature in enumerate(features_to_check, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[feature], color='lightblue')
    plt.title(f'Boxplot of {feature}')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
```

---

## 5. Apply Regression Models

**Theory:** Regression analysis is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables. 

**Practical Implementation:** We will predict `Global_BoxOfficeUSD` using continuous features like `BudgetUSD`, `Opening_Day_SalesUSD`, `IMDbRating`, and `NumVotesIMDb`. We train a Random Forest Regressor and a Linear Regressor.

### Code: Prepare Data for Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Drop rows with missing values in critical columns
reg_features = ['BudgetUSD', 'Opening_Day_SalesUSD', 'IMDbRating', 'NumVotesIMDb']
target = 'Global_BoxOfficeUSD'

reg_df = df.dropna(subset=reg_features + [target])

X_reg = reg_df[reg_features]
y_reg = reg_df[target]

# Split the data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print(f"Training set size: {X_train_r.shape[0]}")
print(f"Testing set size: {X_test_r.shape[0]}")
```

### Code: Train Linear Regression Model

```python
# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)

# Predict
y_pred_lr = lr_model.predict(X_test_r)
```

### Code: Train Random Forest Regressor

```python
# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

# Predict
y_pred_rf = rf_reg.predict(X_test_r)
```

---

## 6. Evaluate Models using Error Metrics (Regression)

**Theory:** 
- **MAE (Mean Absolute Error):** Average magnitude of errors.
- **RMSE (Root Mean Squared Error):** Standard deviation of the prediction errors. Penalizes large errors.
- **R² (Coefficient of Determination):** Proportion of the variance in the dependent variable that is predictable from the independent variables. Value closer to 1 means a very good fit.

**Practical Implementation:** We calculate these metrics to compare Linear Regression and Random Forest.

### Code: Evaluate Regression Models

```python
def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}\n")

evaluate_regression(y_test_r, y_pred_lr, "Linear Regression")
evaluate_regression(y_test_r, y_pred_rf, "Random Forest Regression")
```

---

## 7. Classification

**Theory:** Classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known.

**Practical Implementation:** We will define a 'Blockbuster' as a movie whose `Global_BoxOfficeUSD` is at least 3 times its `BudgetUSD`. We will predict this binary class (1 for Blockbuster, 0 for Not).

### Code: Create Classification Target

```python
# Create Classification Target: 'Is_Blockbuster' (Global Box Office > 3 * Budget)
clf_df = reg_df.copy()
# We define a hit/blockbuster as global box office > 3 times the budget
clf_df['Is_Blockbuster'] = (clf_df['Global_BoxOfficeUSD'] > 3 * clf_df['BudgetUSD']).astype(int)

# Class distribution
print(clf_df['Is_Blockbuster'].value_counts(normalize=True))

# Features for classification 
clf_features = ['BudgetUSD', 'Opening_Day_SalesUSD', 'IMDbRating', 'NumVotesIMDb']
X_clf = clf_df[clf_features]
y_clf = clf_df['Is_Blockbuster']

# Split Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
```

### Code: Train Classification Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_c, y_train_c)
y_pred_log = log_reg.predict(X_test_c)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rfc = rf_clf.predict(X_test_c)
```

---

## 8. Evaluate Models using Error Metrics (Classification)

**Theory:**
- **Accuracy:** Ratio of correctly predicted observations to the total observations.
- **Precision:** Ratio of correctly predicted positive observations to the total predicted positive observations. Useful when false positives are costly.
- **Recall (Sensitivity):** Ratio of correctly predicted positive observations to the all observations in actual class. Useful when false negatives are costly.
- **F1 Score:** Weighted average of Precision and Recall. Useful when we have an uneven class distribution.

**Practical Implementation:** We evaluate the classifiers to see how well they predict 'Blockbusters'.

### Code: Evaluate Classification Models

```python
def evaluate_classification(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

evaluate_classification(y_test_c, y_pred_log, "Logistic Regression")
evaluate_classification(y_test_c, y_pred_rfc, "Random Forest Classifier")
```

### Code: Confusion Matrix and Classification Report

```python
# Confusion Matrix for Random Forest Classifier
cm = confusion_matrix(y_test_c, y_pred_rfc)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Blockbuster', 'Blockbuster'], yticklabels=['Not Blockbuster', 'Blockbuster'])
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report - Random Forest Classifier:")
print(classification_report(y_test_c, y_pred_rfc, target_names=['Not Blockbuster', 'Blockbuster'], zero_division=0))
```

---

## Conclusion

This notebook connected the theory of Exploratory Data Analysis, Visualization, Regression, and Classification with practical implementation using Python and Scikit-Learn. We successfully investigated the `movies_dataset.csv`, engineered required features, incorporated actors' popularity representations like `LeadActor` and `NumVotesIMDb`, built models to forecast revenue and categorize blockbusters, all while critically evaluating these models using proper statistical error metrics.
