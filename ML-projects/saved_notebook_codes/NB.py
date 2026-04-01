import pandas as pd
email_class_df = pd.read_csv('../data/new datasset + classifications/spam_Emails_data.csv')

email_class_df.describe()
email_class_df.info()
email_class_df.head(5)

# drop null text
email_class_df = email_class_df.dropna()
email_class_df.describe()
email_class_df.info()

# see distinct labels email class
email_class = email_class_df['label'].value_counts()
print(email_class)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = email_class_df['text'], email_class_df['label'] # Separate text and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')), # Tokenize and count
    ('tfidf', TfidfTransformer()),                   # Step 2: Apply TF-IDF weighting
    ('clf', MultinomialNB(alpha=1.0))                # Step 3: Classifier with Laplace smoothing
])

# Train the entire Pipeline
# You only need to call .fit() once; it handles all three steps internally
text_clf.fit(X_train, y_train)

# Predict and Evaluate
y_pred = text_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# DETAILED EVALUATION

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

email_class = text_clf.classes_  # dynamic email classes

print("--- Basic Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
# pos_label is needed for binary classification metrics
print(f"Precision: {precision_score(y_test, y_pred, pos_label='Spam'):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='Spam'):.2f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, pos_label='Spam'):.2f}")

print("\n--- Detailed Classification Report ---")
# Provides breakdown for every class
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=email_class)
print(cm)

# Optional: Visualizing the Confusion Matrix
# This makes it much easier to see where the model is failing
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=email_class, yticklabels=email_class, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Spam Detection Confusion Matrix')
plt.show()


# NOW we will explore GNB


# TODO we will explore and make a project on this
credit_card_transactions_df = pd.read_csv('../data/new datasset + classifications/credit_card_transactions.csv')

credit_card_transactions_df.describe()
credit_card_transactions_df.info()
credit_card_transactions_df.head(5)


heart_attack_prediction_indonesia_df = pd.read_csv('../data/new datasset + classifications/heart_attack_prediction_indonesia_df.csv')

heart_attack_prediction_indonesia_df.describe()
heart_attack_prediction_indonesia_df.info()
heart_attack_prediction_indonesia_df.head(5)

# alcohol_consumption we will replace it with NO
clean_heart_attack_prediction_indonesia_df = heart_attack_prediction_indonesia_df.copy()
clean_heart_attack_prediction_indonesia_df['alcohol_consumption'] = clean_heart_attack_prediction_indonesia_df['alcohol_consumption'].fillna('No')
clean_heart_attack_prediction_indonesia_df.describe()
clean_heart_attack_prediction_indonesia_df.info()

# first we will apply GNB blindly on the heart-attack prediction data
# later on we will rectify it

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, PowerTransformer
from sklearn.naive_bayes import GaussianNB 

# Preprocessing & Feature Engineering
# - KBinsDiscretizer: Turns continuous numbers into 3 discrete bins (0, 1, 2)
# FIRST we will go with power transformer then we can think of doing KBinsDiscretizer
# - OrdinalEncoder: Turns text categories into unique integers (0, 1, 2)
# we leave some boolean encded variables like obesity, participated_in_free_screening etc.
preprocessor = ColumnTransformer(
    transformers=[
        ('num_bins', PowerTransformer(method='yeo-johnson'),
        ['cholesterol_level', 'waist_circumference', 'sleep_hours', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'fasting_blood_sugar', 
        'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides']),
        #('num_bins', KBinsDiscretizer(n_bins=3, encode='ordinal'),
        #['cholesterol_level', 'waist_circumference', 'sleep_hours', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'fasting_blood_sugar', 
        #'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides']),
        ('cat_ord', OrdinalEncoder(), ['gender', 'region', 'income_level', 'smoking_status', 'alcohol_consumption',
                                        'physical_activity', 'dietary_habits', 'air_pollution_exposure', 'stress_level', 'EKG_results'])
    ])

# Unified GaussianNB Pipeline
# GNB will treat the resulting integers as points on a distribution
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', GaussianNB())
])

# Train the model
X = clean_heart_attack_prediction_indonesia_df.drop('heart_attack', axis=1)
y = clean_heart_attack_prediction_indonesia_df['heart_attack']

# data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the entire Pipeline
# You only need to call .fit() once; it handles all three steps internally
pipeline.fit(X_train, y_train)

# Predict and Evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# DETAILED EVALUATION
target_class = pipeline.classes_  # dynamic email classes

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
plt.title('Heart-Attack Detection Confusion Matrix')
plt.show()