# saved code snippet from notebook

import pandas as pd
jobsalary_df = pd.read_csv('../data/job_salary_prediction_dataset.csv', nrows=100)

jobsalary_df.describe()
jobsalary_df.info()
jobsalary_df.head(5)

from functools import partial
get_ipython().showtraceback = partial(get_ipython().showtraceback, exception_only=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 'salary' is target
X = jobsalary_df.drop('salary', axis=1)
y = jobsalary_df['salary']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model (This might fail if you have NaNs or non-numeric data)
model = LinearRegression()
model.fit(X_train, y_train)

# results
y_pred = model.predict(X_test)
print(f"Initial R square: {r2_score(y_test, y_pred)}")


print(jobsalary_df['job_title'].value_counts())
print(jobsalary_df['industry'].value_counts())
print(jobsalary_df['location'].value_counts())

# transforming and traning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import r2_score

# Ordinal Order
degree_order = [['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']]

# column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('target', TargetEncoder(target_type='continuous'), ['job_title', 'industry', 'location']),
        ('ohe', OneHotEncoder(drop='first'), ['company_size', 'remote_work']),
        ('order', OrdinalEncoder(categories=degree_order), ['education_level'])
    ],
    remainder='passthrough'
)

# bundles the preprocessing AND the model together
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# data spliting for traning and testing
X = jobsalary_df.drop('salary', axis=1)
y = jobsalary_df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fits all encoders AND the model correctly
model_pipeline.fit(X_train, y_train)

# Predict
# the pipeline handles the encoding internally
y_pred = model_pipeline.predict(X_test)
#print(f"Predictions: {predictions}")

# R-squared
r2 = r2_score(y_test, y_pred)

# Adjusted R-squared
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# Extract feature names and coefficients
features = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coeffs = model_pipeline.named_steps['regressor'].coef_

# Create a 'Drivers' table
drivers = pd.DataFrame({'Feature': features, 'Coefficient': coeffs})
drivers['Abs_Influence'] = drivers['Coefficient'].abs()
drivers = drivers.sort_values(by='Abs_Influence', ascending=False)

print(drivers.head(10))

# cross vaiidation score
from sklearn.model_selection import cross_val_score

# Use the whole pipeline
scores = cross_val_score(model_pipeline, X, y, cv=5)
print(f"Average CV R2: {scores.mean():.2f}")

# residual plot
import matplotlib.pyplot as plt
y_pred = model_pipeline.predict(X_test)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.1) # alpha helps see density in 50k points
plt.axhline(0, color='red')
plt.show()

# fits all encoders AND the model correctly
# note: this time we do salary log transformation
y_train_log = np.log1p(y_train)
model_pipeline.fit(X_train, y_train_log)

# Predict
# the pipeline handles the encoding internally
y_pred = model_pipeline.predict(X_test)
#print(f"Predictions: {predictions}")

# also need to transform test data into log
y_test_log = np.log1p(y_test)
# R-squared
r2 = r2_score(y_test_log, y_pred)

# Adjusted R-squared
n = len(y_test_log)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# residual plot 
residuals = y_test_log - y_pred
plt.scatter(y_pred, residuals, alpha=0.1) # alpha helps see density in 50k points
plt.axhline(0, color='red')
plt.show()

import numpy as np

# note: this time we will go with 90:10 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# fits all encoders AND the model correctly
# note: this time we do salary log transformation
y_train_log = np.log1p(y_train)
model_pipeline.fit(X_train, y_train_log)

# Predict
# the pipeline handles the encoding internally
y_pred = model_pipeline.predict(X_test)
#print(f"Predictions: {predictions}")

# also need to transform test data into log
y_test_log = np.log1p(y_test)
# R-squared
r2 = r2_score(y_test_log, y_pred)

# Adjusted R-squared
n = len(y_test_log)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# residual plot 
residuals = y_test_log - y_pred
plt.scatter(y_pred, residuals, alpha=0.1) # alpha helps see density in 50k points
plt.axhline(0, color='red')
plt.show()

import statsmodels.api as sm

# note: this time we will compute p values
# Transform your Training Data using the pipeline's preprocessor
# This handles your TargetEncoding, OneHot, and Ordinal steps
X_train_transformed = model_pipeline.named_steps['preprocessor'].transform(X_train)

# Get the actual feature names (so the table is readable)
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Convert to a DataFrame for statsmodels
X_train_final = pd.DataFrame(X_train_transformed, columns=feature_names)

# Add the 'Intercept' (statsmodels doesn't add it automatically)
X_train_final = sm.add_constant(X_train_final)

# Log-transform your target (since your model uses log(y))
y_train_log = np.log1p(y_train)

# Reset the index of X_train_final and y_train_log to match
y_train_log = np.log1p(y_train).reset_index(drop=True)
X_train_final = X_train_final.reset_index(drop=True)

# Fit the Ordinary Least Squares (OLS) model
sm_model = sm.OLS(y_train_log, X_train_final).fit()

# Print the beautiful summary table
print(sm_model.summary())


# NOTE: now we will drop a column (industry) which is not a great predictor
# column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('target', TargetEncoder(target_type='continuous'), ['job_title', 'location']),
        ('ohe', OneHotEncoder(drop='first'), ['company_size', 'remote_work']),
        ('order', OrdinalEncoder(categories=degree_order), ['education_level'])
    ],
    remainder='passthrough'
)

# bundles the preprocessing AND the model together
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# NOTE THIS: here I am modifying dataframe to prepare dataset

# data spliting for traning and testing
X = jobsalary_df.drop(['salary', 'industry'], axis=1)
y = jobsalary_df['salary']
# 90: 10 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# fits all encoders AND the model correctly
# note: this time we do salary log transformation
y_train_log = np.log1p(y_train)
model_pipeline.fit(X_train, y_train_log)

# Predict
# the pipeline handles the encoding internally
y_pred = model_pipeline.predict(X_test)
#print(f"Predictions: {predictions}")

# also need to transform test data into log
y_test_log = np.log1p(y_test)
# R-squared
r2 = r2_score(y_test_log, y_pred)

# Adjusted R-squared
n = len(y_test_log)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# residual plot 
residuals = y_test_log - y_pred
plt.scatter(y_pred, residuals, alpha=0.1) # alpha helps see density in 50k points
plt.axhline(0, color='red')
plt.show()

# Extract feature names and coefficients
features = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coeffs = model_pipeline.named_steps['regressor'].coef_

# Create a 'Drivers' table
drivers = pd.DataFrame({'Feature': features, 'Coefficient': coeffs})
drivers['Abs_Influence'] = drivers['Coefficient'].abs()
drivers = drivers.sort_values(by='Abs_Influence', ascending=False)

print(drivers.head(10))

# NOTE THIS: here I am modifying dataframe to prepare dataset
# also now deletitng no remote work no

# data spliting for traning and testing
cleaned_jobsalary_df = jobsalary_df[jobsalary_df['remote_work'] == 'No']
X = cleaned_jobsalary_df.drop(['salary', 'industry'], axis=1)
y = cleaned_jobsalary_df['salary']
# 90: 10 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# fits all encoders AND the model correctly
# note: this time we do salary log transformation
y_train_log = np.log1p(y_train)
model_pipeline.fit(X_train, y_train_log)

# Predict
# the pipeline handles the encoding internally
y_pred = model_pipeline.predict(X_test)
#print(f"Predictions: {predictions}")

# also need to transform test data into log
y_test_log = np.log1p(y_test)
# R-squared
r2 = r2_score(y_test_log, y_pred)

# Adjusted R-squared
n = len(y_test_log)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# residual plot 
residuals = y_test_log - y_pred
plt.scatter(y_pred, residuals, alpha=0.1) # alpha helps see density in 50k points
plt.axhline(0, color='red')
plt.show()

# Extract feature names and coefficients
features = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coeffs = model_pipeline.named_steps['regressor'].coef_

# Create a 'Drivers' table
drivers = pd.DataFrame({'Feature': features, 'Coefficient': coeffs})
drivers['Abs_Influence'] = drivers['Coefficient'].abs()
drivers = drivers.sort_values(by='Abs_Influence', ascending=False)

print(drivers.head(10))