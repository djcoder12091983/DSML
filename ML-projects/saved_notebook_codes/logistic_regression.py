import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import LabelEncoder

# target
y = hdprediction_df['Thallium']

# X_num: Numeric features | X_cat: Categorical features
X_num = hdprediction_df[['Age', 'BP', 'Cholesterol', 'Max HR', 'Number of vessels fluro', 'ST depression', 'Slope of ST']]
X_cat = hdprediction_df[['Sex', 'Chest pain type', 'FBS over 120', 'EKG results']]

# Test Numeric Predictors (ANOVA)
num_selector = SelectKBest(score_func=f_classif, k='all')
X_num_best = num_selector.fit_transform(X_num, y)

# Get scores (Higher is better)
num_scores = pd.DataFrame({
    'Feature': X_num.columns,
    'ANOVA_Score': num_selector.scores_,
    'P_Value': num_selector.pvalues_
}).sort_values(by='ANOVA_Score', ascending=False)

# Test: Chi-square
cat_selector = SelectKBest(score_func=chi2, k='all')
X_cat_best = cat_selector.fit_transform(X_cat, y)

cat_scores = pd.DataFrame({
    'Feature': X_cat.columns,
    'Chi2_Score': cat_selector.scores_,
    'P_Value': cat_selector.pvalues_
}).sort_values(by='Chi2_Score', ascending=False)

print("Numeric Predictors: ", num_scores)
print("Categorical Predictors: ", cat_scores)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Target 'y' should stay as integer labels (0, 1, 2)
target_le = LabelEncoder()
y = target_le.fit_transform(hdprediction_df['Thallium'])
# input data
X = hdprediction_df.drop(['id', 'BP', 'Cholesterol', 'FBS over 120', 'Thallium'], axis=1)
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup Features
numeric_features = ['Age', 'Max HR', 'Number of vessels fluro', 'ST depression', 'Slope of ST']
categorical_features = ['Sex', 'Chest pain type', 'EKG results']

# Define Column Transformations
# Scaling for numbers; OHE for numeric-encoded categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Build the Pipeline, NOTE: SAGA to handle large data
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='saga', 
        max_iter=1000, 
        random_state=42
        ))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()


# Extract the model from your pipeline
model = pipeline.named_steps['classifier']
# Get features names from your preprocessor/dataframe
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Plot weights for Class 1 (Fixed Defect) - where your precision was 0.0
# coef_[1] refers to the weights for the second class
plt.figure(figsize=(10, 5))
plt.barh(feature_names, model.coef_[1], color='teal')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('How Features Influence "Fixed Defect" Predictions')
plt.xlabel('Coefficient Value (Importance)')
plt.show()


# NOTE: This time handling IMBALANCED data
# Build the Pipeline, NOTE: SAGA to handle large data
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=1000, 
        random_state=42
        ))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()


# NOTE: This time handling IMBALANCED data by some tolerance set 0.01
# Build the Pipeline, NOTE: SAGA to handle large data
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=3000,
        tol=1e-2,
        # optimal CPU usage
        n_jobs=-1,
        random_state=42
        ))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# NOTE: This time handling IMBALANCED data by some tolerance set 0.01 and custom weights {0:1, 1:5, 2:1}
# Build the Pipeline, NOTE: SAGA to handle large data
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='saga',
        class_weight={0:1, 1:5, 2:1},
        max_iter=3000,
        tol=1e-2,
        # optimal CPU usage
        n_jobs=-1,
        random_state=42
        ))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# Sampling Strategy
# We keep all 4k of Class 1, and take 25k each of Class 0 and 2
sampling_strategy = {0: 25000, 1: 4000, 2: 25000}

# NOTE: This time handling IMBALANCED data with undersampling
# Build the Pipeline, NOTE: still giving a try with SAGA solver
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('undersampler', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
    ('classifier', LogisticRegression(solver='saga', max_iter=2000, tol=1e-3, random_state=42))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Sampling Strategy
# We keep all 4k of Class 1, and take 25k each of Class 0 and 2
sampling_strategy = {0: 25000, 1: 4000, 2: 25000}

# NOTE: This time handling IMBALANCED data with undersampling
# Build the Pipeline, NOTE: still giving a try with SAGA solver
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('undersampler', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
    ('classifier', LogisticRegression(
                solver='saga',
                # this time manual assigning weight
                class_weight={0:1, 1:50, 2:1},
                max_iter=2000,
                tol=1e-3,
                random_state=42
            ))
])

# Fit the Model
pipeline.fit(X_train, y_train)

# Evaluation
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test) # result

# Detailed breakdown per class (Normal, Fixed, Reversible)
print(classification_report(y_test, y_pred))

# See the 'Misses' vs 'Hits'
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Overall ability to distinguish classes (Target: > 0.80)
# For multiclass, use 'ovr' or 'ovo' for AUC
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Compute the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with Seaborn for better styling
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2'],
            yticklabels=['0', '1', '2'])

plt.title('Heart Disease Classification: Error Tracking')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Compare the mean of a key feature like 'oldpeak' or 'max_heart_rate'
print(hdprediction_df.groupby('Thallium')[['Age', 'Max HR', 'Number of vessels fluro', 'Slope of ST']].mean())


# List of features with similar means
features_to_check = ['Age', 'Max HR', 'Number of vessels fluro', 'Slope of ST']

# Create a subplot for each feature
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(features_to_check):
    if i < len(axes):
        sns.kdeplot(data=hdprediction_df, x=col, hue='Thallium', fill=True, 
                    palette='bright', common_norm=False, ax=axes[i])
        axes[i].set_title(f'Distribution of {col} across Groups')

plt.tight_layout()
plt.show()

# TODO may need to fix issues with imbalanced data