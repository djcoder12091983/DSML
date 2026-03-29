# IRIS data
import pandas as pd
iris_df = pd.read_csv('../data/iris.csv')

iris_df.describe()
iris_df.info()
iris_df.head(5)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the Iris Dataset
X = iris_df.drop('Species', axis = 1)  # Features: sepal/petal length and width
# target variable label encoding
y = iris_df['Species']  # Target: species (0, 1, or 2)
le = LabelEncoder()
y = le.fit_transform(y)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling (Essential for distance-based models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# n_neighbors=3 means we look at the 3 closest points
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make Predictions
predictions = knn.predict(X_test)

# Evaluate Performance
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, predictions))


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# visualization of confusion matrix
plt.figure(figsize=(8, 6))
# Using ConfusionMatrixDisplay for a professional look
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix: Perfect Classification')
plt.show()


# plot elbow curve
# Calculate error rates for a range of K values
error_rate = []
for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # Average of where predicted label does not match actual label
    error_rate.append(np.mean(pred_i != y_test))

# Plot the Error Rate vs K Value
plt.figure(figsize=(10, 6))
plt.plot(range(1, 41), error_rate, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Elbow Method)')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

import seaborn as sns

# KDE plots to check how IRIS data is perfect meaning they are not overlapping
cols_to_plot = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes = axes.flatten()

# draw each plot
for i, col in enumerate(cols_to_plot):
    sns.kdeplot(data=iris_df, x=col, ax=axes[i], hue='Species', fill=True)
    
    # Customizing each subplot
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')

plt.tight_layout() # Fixes overlapping titles/labels
plt.show()

# Pima Indians Diabetes data
import pandas as pd
Pima_Diabetes_df = pd.read_csv('../data/Pima Indians Diabetes.csv')

Pima_Diabetes_df.describe()
Pima_Diabetes_df.info()
Pima_Diabetes_df.head(5)

# Load the Dataset
X = Pima_Diabetes_df.drop('Outcome', axis = 1)
# target variable label encoding
y = Pima_Diabetes_df['Outcome']
le = LabelEncoder()
y = le.fit_transform(y)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling (Essential for distance-based models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# n_neighbors=3 means we look at the 3 closest points
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make Predictions
predictions = knn.predict(X_test)

# Evaluate Performance
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, predictions))

# visualization of confusion matrix
plt.figure(figsize=(8, 6))
# Using ConfusionMatrixDisplay for a professional look
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix')
plt.show()

# KDE plots to check how data is overlapping
cols_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
axes = axes.flatten()

# draw each plot
for i, col in enumerate(cols_to_plot):
    sns.kdeplot(data=Pima_Diabetes_df, x=col, ax=axes[i], hue='Outcome', fill=True)
    
    # Customizing each subplot
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')

plt.tight_layout() # Fixes overlapping titles/labels
plt.show()

# plot elbow curve
# Calculate error rates for a range of K values
error_rate = []
for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # Average of where predicted label does not match actual label
    error_rate.append(np.mean(pred_i != y_test))

# Plot the Error Rate vs K Value
plt.figure(figsize=(10, 6))
plt.plot(range(1, 41), error_rate, color='blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Elbow Method)')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# filter columns by 0 counts
label_counts = {}
columns = Pima_Diabetes_df.columns.drop('Outcome')
for col in columns:
    data = Pima_Diabetes_df[col]
    count = data[data == 0].count()
    if count > 0:
        label_counts[col] = count
        
labels = list(label_counts.keys())
counts = list(label_counts.values())

# Generate Random Color Combinations
# We generate a unique RGB tuple for each label
np.random.seed(42) # Optional: Remove this for "true" randomness every time
random_colors = np.random.rand(len(labels), 3) 

# Create the Pie Chart
plt.figure(figsize=(10, 7))

# Parameters explained:
# x: The counts/values
# labels: The names of the categories
# autopct: String format to show both the percentage and the raw count
# colors: Our generated random RGB array
# startangle: Rotates the chart for better alignment
patches, texts, autotexts = plt.pie(
    counts, 
    labels=labels, 
    autopct=lambda p: '{:.1f}%\n({:d} units)'.format(p, int(p/100.*sum(counts))),
    colors=random_colors,
    startangle=140,
    shadow=True,
    wedgeprops={'edgecolor': 'black'} # Adds a border to separate random colors
)

# Styling the text for readability
plt.setp(autotexts, size=10, weight="bold", color="white")
plt.title("Filter columns by zero count", fontsize=15, pad=20)

plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()


# BAR charts
# Extract keys (labels) and values (counts) for plotting
labels = list(label_counts.keys())
counts = list(label_counts.values())

# Generate Random Colors for each bar
# We use a colormap 'hsv' to get a diverse range of colors automatically
colors = plt.cm.hsv(np.linspace(0, 1, len(labels)))

# Create the Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.2)

# Add Data Labels (The exact count on top of each bar)
# This is a professional touch so you don't have to "guess" the Y-axis value
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Final Styling
ax.set_title('Filter columns by zero count', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Zero counts', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7) # Grid only on Y-axis for scale

plt.tight_layout()
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

# now we will apply some clean logic and then apply KNN again and evaluate again
# Pre-process: Replace 0 with NaN
# (This must happen before the pipeline because it's a data-cleaning step)
Pima_Diabetes_clean_df = Pima_Diabetes_df.copy()
for col in label_counts.keys():
    Pima_Diabetes_clean_df[col] = Pima_Diabetes_clean_df[col].replace(0, np.nan)

X = Pima_Diabetes_clean_df.drop('Outcome', axis=1)
y = Pima_Diabetes_clean_df['Outcome']

# Build the Integrated Pipeline
# Order: Scale -> Impute -> Classify
# We scale first so the KNNImputer uses "fair" distances to find neighbors
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Standardize features
    ('imputer', KNNImputer(n_neighbors=2)), # Fill NaNs using neighbors
    ('classifier', KNeighborsClassifier(n_neighbors=3)) # Final Model
])

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The pipeline handles everything!
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# To see the imputed/scaled data for debugging:
# X_processed = pipeline.named_steps['scaler'].transform(X_train)
# X_imputed = pipeline.named_steps['imputer'].transform(X_processed)

#print("Final Processed Data (Scaled & Imputed):\n", X_imputed)

# Evaluate Performance
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, predictions))

# visualization of confusion matrix
plt.figure(figsize=(8, 6))
# Using ConfusionMatrixDisplay for a professional look
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix after Data Imputation')
plt.show()



# this time we will try to find optimal hyperparameters using grid search
from sklearn.model_selection import GridSearchCV

# Select Range of Values
param_grid = {
    # Odd numbers to avoid ties
    'imputer__n_neighbors': [3, 5, 7, 9, 11, 15],
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    
    'classifier__weights': ['uniform', 'distance'],   # Weighting by distance or not
    'classifier__metric': ['euclidean', 'manhattan']  # Try different distance math
}

# Run Grid Search
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# EVALUATION (The most important part)
# Use the BEST model found by grid search to predict on the UNSEEN test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Detailed Metrics
print(f"Best Parameters: {grid.best_params_}")
print("\n--- Evaluation Metrics ---")
print(classification_report(y_test, y_pred))

# Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix of Best KNN Model")
plt.show()