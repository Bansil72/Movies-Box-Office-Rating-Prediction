# Overview of `box_office_rating_eda.ipynb`

This notebook performs a complete exploratory data analysis (EDA) and predictive modeling workflow on the `movies_dataset.csv` dataset.

## What the notebook contains

1. **Data loading and inspection**
   - Imports essential Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn tools.
   - Loads the dataset and prints the shape.
   - Shows the first few rows using `df.head()`.
   - Displays dataset information, descriptive statistics, and missing value counts.

2. **Exploratory Data Analysis (EDA)**
   - Builds univariate visualizations for budget and global box office distributions.
   - Plots movie counts by genre.
   - Explores bivariate relationships, including:
     - Top lead actors by average global box office.
     - IMDb vote counts versus global box office.
     - Global box office distribution by genre with boxplots.
   - Computes a correlation matrix for numeric features.
   - Uses a genre-colored scatter plot to compare budget versus box office performance.

3. **Data cleaning**
   - Drops duplicate rows.
   - Imputes missing values:
     - Numeric columns with median values.
     - Categorical columns with mode values.

4. **Outlier detection**
   - Creates boxplots for numerical features such as:
     - `BudgetUSD`
     - `Global_BoxOfficeUSD`
     - `Opening_Day_SalesUSD`
     - `One_Week_SalesUSD`
     - `NumVotesIMDb`
     - `NumVotesRT`
   - Helps identify potential outliers before modeling.

5. **Regression modeling**
   - Prepares regression data using features:
     - `BudgetUSD`
     - `Opening_Day_SalesUSD`
     - `IMDbRating`
     - `NumVotesIMDb`
   - Predicts `Global_BoxOfficeUSD` using:
     - Linear Regression
     - Random Forest Regression

6. **Regression evaluation**
   - Measures model performance with:
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R² score
   - Compares the two regression models based on these metrics.

7. **Classification modeling**
   - Creates a binary target: `Is_Blockbuster`.
   - Defines blockbusters as movies with global box office greater than three times the budget.
   - Trains classification models using:
     - Logistic Regression
     - Random Forest Classifier

8. **Classification evaluation**
   - Evaluates classifiers using:
     - Accuracy
     - Precision
     - Recall
     - F1 score
   - Shows confusion matrix and classification report for the Random Forest Classifier.

## Purpose and value

This notebook is designed to:
- Explore the movie dataset with strong visual and statistical analysis.
- Clean the dataset for further modeling and improve data quality.
- Build and compare regression and classification models for revenue prediction and blockbuster detection.
- Demonstrate a full EDA-to-modeling workflow using Python, pandas, seaborn, and scikit-learn.
- Provide a reusable example for movie industry analytics and revenue forecasting.

## Dataset insights and assumptions

The notebook assumes the dataset includes columns for:
- `BudgetUSD`: Production budget in US dollars.
- `Global_BoxOfficeUSD`: Worldwide box office revenue.
- `Opening_Day_SalesUSD` and `One_Week_SalesUSD`: Early revenue indicators.
- `IMDbRating`: Average audience rating on IMDb.
- `NumVotesIMDb` and `NumVotesRT`: Popularity and visibility metrics.
- `Genre` and `LeadActor`: Categorical variables used for exploratory comparisons.

It also assumes the dataset may have:
- Duplicate rows that should be removed.
- Missing values that require imputation.
- Numeric outliers that may affect modeling performance.

## Expected output from the notebook

Running this notebook produces:
- Summary statistics and data quality checks.
- Visual plots for distributions, genre comparisons, and feature relationships.
- A correlation heatmap that highlights strong and weak relationships among numeric variables.
- Regression model results showing how closely budget and popularity features can predict box office revenue.
- Classification model results identifying which films qualify as blockbusters.
- A confusion matrix and classification report for the best classifier.

## Recommended next steps

After running the notebook, you can extend it by:
- Adding more features such as `Runtime`, `ProductionCompany`, or `ReleaseYear`.
- Engineering new variables like budget-to-rating ratio, release quarter, or actor popularity score.
- Comparing alternative models such as XGBoost, Gradient Boosting, or Support Vector Machines.
- Using cross-validation to improve model selection and reduce overfitting.
- Creating an interactive dashboard or report using Streamlit or Plotly.

## How to use it

- Run the notebook from top to bottom.
- Ensure `movies_dataset.csv` is available in the same folder.
- Inspect outputs after each section to understand data quality and model performance.
- Use the generated plots and metrics to inform improvements or further feature engineering.
- Review the regression and classification metric outputs to decide which model is strongest for your dataset.
