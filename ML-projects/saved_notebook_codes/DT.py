# DT and random forset exploration with NYC airbnb data
import pandas as pd
airbnb_df = pd.read_csv('../data/AB_NYC_2019.csv')

airbnb_df.describe()
airbnb_df.info()
airbnb_df.head(5)

# NULL count per columns
na_counts = {}
columns = airbnb_df.columns
for col in columns:
    data = airbnb_df[col]
    count = data[data.isna() | data.isnull()].size
    if count > 0:
        na_counts[col] = count
        
print(na_counts)

# let's take another data NYC housing price
housingprice_df = pd.read_csv('../data/nyc_housing_base.csv')

housingprice_df.describe()
housingprice_df.info()
housingprice_df.head(5)

# NULL count per columns
na_counts = {}
columns = housingprice_df.columns
for col in columns:
    data = housingprice_df[col]
    count = data[data.isna() | data.isnull()].size
    if count > 0:
        na_counts[col] = count
        
print(na_counts)

# find category variables frequency count
cat_var = ['borough_x', 'zip_code', 'borough_y', 'landuse', 'bldgclass']
for col in cat_var:
    print(col)
    print('=====================')
    print(housingprice_df.groupby(col).size())
    
# Forward Fill for Categorical/numerical (geographic) Columns
ffill_cols = ['zip_code', 'latitude', 'longitude', 'landuse']
clean_housingprice_df = housingprice_df.copy()
clean_housingprice_df[ffill_cols] = clean_housingprice_df[ffill_cols].ffill()

# NULL count per columns
na_counts = {}
columns = clean_housingprice_df.columns
for col in columns:
    data = clean_housingprice_df[col]
    count = data[data.isna() | data.isnull()].size
    if count > 0:
        na_counts[col] = count
        
print(na_counts)


# straight forward decision with full columns after doing some cleaning and transformation
# let's see how it goes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import TargetEncoder

# Identify Columns
# num_cols = na_counts.keys() # now remaining columns having null/NA values will be part of mean imputer
num_cols = ['resarea', 'comarea', 'numfloors']
cat_cols = ['borough_x', 'zip_code', 'borough_y', 'landuse', 'bldgclass']

# Define Preprocessing Steps

# Mean Imputation
num_transformer = SimpleImputer(strategy='mean')

# Categorical: Target Encoding
cat_transformer = TargetEncoder() 

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
# Apply these 'Speed Limits' to your Regressor
model = DecisionTreeRegressor(
    max_depth=10,            # Stop the tree from growing to infinity
    min_samples_leaf=50,     # Don't create a split unless there are 50 rows
    max_features='sqrt',     # Look at fewer features per split to speed up
    random_state=42
)

# This pipeline: 1. Imputes/Encodes -> 2. Fits Decision Tree
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
    
    ('regressor', model)
])

# Split Data
X = clean_housingprice_df.drop('sale_price', axis=1)
y = clean_housingprice_df['sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The 'y_train' is automatically passed to the TargetEncoder during fit!
pipeline.fit(X_train, y_train)

# Predict & Evaluate
# When predicting, the pipeline uses the MEANS calculated during training
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model trained successfully!")
print(f"Test RMSE: {rmse:.2f}")


# check target value distribution and skewness
import seaborn as sns
import matplotlib.pyplot as plt

# Create a horizontal boxplot for the 'price' column
plt.figure(figsize=(10, 5))
sns.boxplot(x=clean_housingprice_df['sale_price'], color='salmon')

# Adding labels and title
plt.title("Boxplot of Target Variable (House Price)")
plt.xlabel("Price")
plt.show()

# Visualize the Raw Distribution
plt.figure(figsize=(10, 6))
sns.histplot(clean_housingprice_df['sale_price'], kde=True, color='blue', bins=50)

plt.title(f"Target Distribution (Skewness: {clean_housingprice_df['sale_price'].skew():.2f})")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# Check for Log-Transformation (If skewness is high > 1)
import numpy as np
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(clean_housingprice_df['sale_price']), kde=True, color='green', bins=50)

plt.title("Log-Transformed Target Distribution")
plt.xlabel("Log(Price + 1)")
plt.show()


from sklearn.compose import TransformedTargetRegressor

# THIS TIME we will apply log transformation along with median numeric IMPUTER

# Imputation
num_transformer = SimpleImputer(strategy='median') # median numeric IMPUTER

# THIS TIME we will apply log transformation
model_with_log = TransformedTargetRegressor(
    regressor=pipeline, # pipeline remains same
    
    # Wrap the Pipeline in a Target Transformer
    # func=np.log1p applies log(1+x) to y before fitting
    # inverse_func=np.expm1 applies exp(x)-1 to predictions automatically
    func=np.log1p, 
    inverse_func=np.expm1
)

# Internally: y_train is converted to log scale, then the tree is trained
model_with_log.fit(X_train, y_train)

# Predict & Evaluate
# When predicting, the pipeline uses the MEANS calculated during training
y_pred = model_with_log.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model trained successfully!")
print(f"Test RMSE: {rmse:.2f}")


# NOW we will back to full length features decision tree nodel with log transformation traget values

model = DecisionTreeRegressor(
    max_depth=10,            # Stop the tree from growing to infinity
    random_state=42
)

# Internally: y_train is converted to log scale, then the tree is trained
model_with_log.fit(X_train, y_train)

# Predict & Evaluate
# When predicting, the pipeline uses the MEANS calculated during training
y_pred = model_with_log.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model trained successfully!")
print(f"Test RMSE: {rmse:.2f}")

# error percentage
error_percentage = rmse/clean_housingprice_df['sale_price'].mean() * 100
print(f"Error percentage:  {error_percentage:0.3f}")

from sklearn.metrics import r2_score

# r2 check
r2 = r2_score(y_test, y_pred)
print(f"R square: {r2:0.4f}")

# individual error check
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)


# THIS time we will try to find important features

# Access the preprocessor from your inner pipeline
# (Adjust 'preprocessor' if you named that step differently in your Pipeline)
preprocessor = model_with_log.regressor_.named_steps['preprocessor']

# Get the feature names after transformation
# This is the "magic" function in modern Scikit-Learn
feature_names = preprocessor.get_feature_names_out()

# Access the actual tree model
regressor = model_with_log.regressor_.named_steps['regressor']

# Combine them into a Series
importances = pd.Series(regressor.feature_importances_, index=feature_names)

# Sort and display
print("Feature Importances (Sorted):")
print(importances.sort_values(ascending=False))
print("Selected Features:")
selected_columns = list(map(lambda x: x[5:], feature_names.tolist()))
print(selected_columns)


# THIS time we will train on selected features

# Identify selected Columns
num_cols = [col for col in num_cols if col in selected_columns]
cat_cols = [col for col in cat_cols if col in selected_columns]

# Categorical: Target Encoding with smoothing
cat_transformer = TargetEncoder(smooth=50)

# THIS TIME we will apply log transformation
model_with_log = TransformedTargetRegressor(
    regressor=pipeline, # pipeline remains same
    
    # Wrap the Pipeline in a Target Transformer
    # func=np.log1p applies log(1+x) to y before fitting
    # inverse_func=np.expm1 applies exp(x)-1 to predictions automatically
    func=np.log1p, 
    inverse_func=np.expm1
)

# Split Data
X = clean_housingprice_df[selected_columns]
y = clean_housingprice_df['sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Internally: y_train is converted to log scale, then the tree is trained
model_with_log.fit(X_train, y_train)

# Predict & Evaluate
# When predicting, the pipeline uses the MEANS calculated during training
y_pred = model_with_log.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model trained successfully!")
print(f"Test RMSE: {rmse:.2f}")

# error percentage
error_percentage = rmse/clean_housingprice_df['sale_price'].mean() * 100
print(f"Error percentage:  {error_percentage:0.3f}")

# r2 check
r2 = r2_score(y_test, y_pred)
print(f"R square: {r2:0.4f}")

# individual error check
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)


# THIS time we will apply random forest

from sklearn.ensemble import RandomForestRegressor

# Define the Random Forest Model
# n_estimators=100: We train 100 trees and average them
# n_jobs=-2: Use CPU cores for speed on local machine or laptop
rf_base = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12,           # Slightly deeper than a single tree is okay
    min_samples_leaf=15,    # Prevents trees from memorizing outliers
    #max_features='sqrt',   # Essential for Random Forest diversity
    max_features = 1.0,     # i have selected all best features
    n_jobs=-2,              # Parallel processing for laptop or local machine
    random_state=42
)

# Wrap in Target Transformation (Log Scale)
model_final = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rf', rf_base)
    ]),
    func=np.log1p, 
    inverse_func=np.expm1
)

# Train the Model
model_final.fit(X_train, y_train)

# Evaluate
y_pred = model_final.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Results:")
print(f"Test RMSE: {rmse:,.2f}")
print(f"R-Squared: {r2:.4f}")

# individual error check
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)


# this time we will apply max_features = 0.5
rf_base = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12,           # Slightly deeper than a single tree is okay
    min_samples_leaf=15,    # Prevents trees from memorizing outliers
    #max_features='sqrt',   # Essential for Random Forest diversity
    max_features = 0.5,     # i have selected all best features, taking half of them randomly
    n_jobs=-2,              # Parallel processing for laptop or local machine
    random_state=42
)

# Train the Model
model_final.fit(X_train, y_train)

# Evaluate
y_pred = model_final.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Results:")
print(f"Test RMSE: {rmse:,.2f}")
print(f"R-Squared: {r2:.4f}")

# individual error check
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)

# this time we will try some higher depth and modified smoothing for target encoding

# Categorical: Target Encoding with smoothing
cat_transformer = TargetEncoder(smooth=80)

rf_base = RandomForestRegressor(
    n_estimators=100, 
    max_depth=40,           # Slightly deeper than a single tree is okay
    min_samples_leaf=5,    # Prevents trees from memorizing outliers
    #max_features='sqrt',   # Essential for Random Forest diversity
    max_features = 1.0,     # i have selected all best features, taking half of them randomly
    n_jobs=-2,              # Parallel processing for laptop or local machine
    random_state=42
)

# Train the Model
model_final.fit(X_train, y_train)

# Evaluate
y_pred = model_final.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Results:")
print(f"Test RMSE: {rmse:,.2f}")
print(f"R-Squared: {r2:.4f}")

# individual error check
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)


# THIS TIME we will try boosting

from xgboost import XGBRegressor

# Define the XGBoost Model
# XGBoost is very powerful; we use 'learning_rate' to prevent overfitting
xgb_base = XGBRegressor(
    n_estimators=1000,     # High number of trees (Boosting handles this well)
    learning_rate=0.05,    # Small steps to "slowly" learn the complex patterns
    max_depth=5,           # XGBoost trees are usually shallower than RF
    subsample=0.8,         # Use 80% of data per tree to prevent memorizing outliers
    colsample_bytree=0.8,  # Similar to max_features='sqrt'
    n_jobs=-2,             # for local machine and laptop
    random_state=42,
    tree_method='hist'     # High-speed method for 35k+ rows
)

# Wrap in Target Transformation (Log Scale)
boosting_model = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', xgb_base)
    ]),
    func=np.log1p, 
    inverse_func=np.expm1
)

# Train the Model
boosting_model.fit(X_train, y_train)

# Evaluate
y_pred = boosting_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"XGBoosting Results:")
print(f"Test RMSE: {rmse:,.2f}")
print(f"R-Squared: {r2:.4f}")

# individual error check
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(mape)