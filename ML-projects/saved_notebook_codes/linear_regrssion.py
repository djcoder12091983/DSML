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