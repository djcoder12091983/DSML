import pandas as pd
covid_df = pd.read_csv('../data/new datasset + classifications/covid19_patient_symptoms_diagnosis.csv')

covid_df.describe()
covid_df.info()
covid_df.head(5)

# data cleaning
clean_covid_df = covid_df.copy()
clean_covid_df['comorbidity'] = clean_covid_df['comorbidity'].fillna('No')

clean_covid_df.describe()
clean_covid_df.info()

# check distinct values
cols = ['comorbidity']
print(clean_covid_df['comorbidity'].value_counts())

# BLIND SVM on all columns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Separate Features and Target
X = clean_covid_df.drop(['patient_id', 'covid_result'], axis=1)
y = clean_covid_df['covid_result']

# Define Preprocessing for Numerical and Categorical Features
numeric_features = ['age', 'oxygen_level', 'body_temperature']
numeric_transformer = StandardScaler()

# other than numerical all are categorical and boolean or having 3-4 distinct values
# so going with oe hot encoding
categorical_features = [col for col in X.columns if col not in numeric_features]
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop = 'first')

# Combine Preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full Pipeline (Preprocessing + SVM Model)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear')) # Linear kernel for binary classification
])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
clf.fit(X_train, y_train)

# Evaluate performance
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# DETAILED EVALUATION
target_class = clf.classes_  # dynamic classes

print("--- Basic Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
# pos_label is needed for binary classification metrics
print(f"Precision: {precision_score(y_test, y_pred, pos_label=1):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label=1):.2f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, pos_label=1):.2f}")

print("\n--- Detailed Classification Report ---")
# Provides breakdown for every class
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=target_class)
print(cm)

# Optional: Visualizing the Confusion Matrix
# This makes it much easier to see where the model is failing
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_class, yticklabels=target_class, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('COVID-19 Detection Confusion Matrix')
plt.show()

# stroke data
stroke_df = pd.read_csv('../data/new datasset + classifications/stroke_risk_dataset.csv')

stroke_df.describe()
stroke_df.info()
stroke_df.head(5)


# BLIND SVM on all columns

# Separate Features and Target
X = stroke_df.drop(['Stroke Risk (%)', 'At Risk (Binary)'], axis=1)
y = stroke_df['At Risk (Binary)']

# Define Preprocessing for Numerical and Categorical Features
numeric_features = ['Age']
numeric_transformer = StandardScaler()

# other than numerical all are categorical and boolean
# so going with oe hot encoding
categorical_features = [col for col in X.columns if col not in numeric_features]
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop = 'first')

# Combine Preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full Pipeline (Preprocessing + SVM Model)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear')) # Linear kernel for binary classification
])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
clf.fit(X_train, y_train)

# Evaluate performance
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# DETAILED EVALUATION
target_class = clf.classes_  # dynamic classes

print("--- Basic Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
# pos_label is needed for binary classification metrics
print(f"Precision: {precision_score(y_test, y_pred, pos_label=1):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label=1):.2f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, pos_label=1):.2f}")

print("\n--- Detailed Classification Report ---")
# Provides breakdown for every class
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=target_class)
print(cm)

# Optional: Visualizing the Confusion Matrix
# This makes it much easier to see where the model is failing
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_class, yticklabels=target_class, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Stroke Detection Confusion Matrix')
plt.show()



from sklearn.model_selection import StratifiedKFold, cross_validate

# NOW we will test on K=10 fold cross validation because my models are showing 100% result
# Separate Features and Target
X = clean_covid_df.drop(['patient_id', 'covid_result'], axis=1)
y = clean_covid_df['covid_result']

# Define Preprocessing for Numerical and Categorical Features
numeric_features = ['age', 'oxygen_level', 'body_temperature']
numeric_transformer = StandardScaler()

# other than numerical all are categorical and boolean or having 3-4 distinct values
# so going with oe hot encoding
categorical_features = [col for col in X.columns if col not in numeric_features]
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop = 'first')

# Combine Preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full Pipeline (Preprocessing + SVM Model)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear')) # Linear kernel for binary classification
])

# CROSS validation
# Define the 10-Fold Stratified Split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the metrics to evaluate
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Run Cross-Validation
results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False)

# Display Detailed Results
print(f"{'Metric':<15} | {'Mean Score':<12} | {'Std Dev'}")
print("-" * 40)
for metric in scoring:
    mean_score = results[f'test_{metric}'].mean()
    std_score = results[f'test_{metric}'].std()
    print(f"{metric.capitalize():<15} | {mean_score:.4f}       | {std_score:.4f}")
    

# NOW we will test on K=10 fold cross validation on different set of data because my models are showing 100% result
# Separate Features and Target
X = stroke_df.drop(['Stroke Risk (%)', 'At Risk (Binary)'], axis=1)
y = stroke_df['At Risk (Binary)']

# Define Preprocessing for Numerical and Categorical Features
numeric_features = ['Age']
numeric_transformer = StandardScaler()

# other than numerical all are categorical and boolean or having 3-4 distinct values
# so going with oe hot encoding
categorical_features = [col for col in X.columns if col not in numeric_features]
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop = 'first')

# Combine Preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full Pipeline (Preprocessing + SVM Model)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear')) # Linear kernel for binary classification
])

# CROSS validation
# Define the 10-Fold Stratified Split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the metrics to evaluate
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Run Cross-Validation
results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False)

# Display Detailed Results
print(f"{'Metric':<15} | {'Mean Score':<12} | {'Std Dev'}")
print("-" * 40)
for metric in scoring:
    mean_score = results[f'test_{metric}'].mean()
    std_score = results[f'test_{metric}'].std()
    print(f"{metric.capitalize():<15} | {mean_score:.4f}       | {std_score:.4f}")