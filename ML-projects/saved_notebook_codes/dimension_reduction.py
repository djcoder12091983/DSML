# we will see on available data on sklearn lib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# Load Data (8x8 images of digits 0-9)
digits = datasets.load_digits(as_frame = True)

# for display purpose
digits_df = digits.frame
digits_df.describe()
digits_df.info()
#digits_df.head(5)

X, y = digits.data, digits.target

# Apply Dimensionality Reduction
pca_res = PCA(n_components=2).fit_transform(X)
lda_res = LDA(n_components=2).fit_transform(X, y) # Requires 'y' labels
tsne_res = TSNE(n_components=2, random_state=42).fit_transform(X)

# Visualize Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['PCA (Max Variance)', 'LDA (Max Class Separation)', 't-SNE (Local Clusters)']
data_list = [pca_res, lda_res, tsne_res]

for ax, data, title in zip(axes, data_list, titles):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax)

plt.show()


from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create a dummy dataset with 100 features
X, y = make_classification(n_samples=500, n_features=100, n_informative=15, random_state=42)

# Standardize the data (Vital for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA to all components to see the full "spectrum"
pca = PCA()
pca.fit(X_scaled)

# Calculate Individual and Cumulative Variance
exp_var_ratio = pca.explained_variance_ratio_
cum_var_ratio = np.cumsum(exp_var_ratio)

# Create the Elbow / Scree Plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, 101), exp_var_ratio, alpha=0.5, label='Individual Variance')
plt.step(range(1, 101), cum_var_ratio, where='mid', label='Cumulative Variance', color='red')

# Add a 90% threshold line
plt.axhline(y=0.90, color='green', linestyle='--', label='90% Threshold')

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Elbow Plot: Finding the Sweet Spot')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Programmatically find the "Sweet Spot" for 90% variance
n_components_90 = np.argmax(cum_var_ratio >= 0.90) + 1
print(f"To keep 90% of the variance, you need {n_components_90} components.")

# we will now see how PCA affects performance

import pandas as pd
heart_disease_df = pd.read_csv('../data/CVD_cleaned.csv')

heart_disease_df.describe()
heart_disease_df.info()
heart_disease_df.head(5)

# check how many category variables are there
for col in heart_disease_df.columns:
    if heart_disease_df[col].dtype == 'str':
        print(f"============{col}============")
        print(heart_disease_df[col].value_counts())
        
# Returns a count of every data type in your DataFrame
print(heart_disease_df.dtypes.value_counts())

cat_cols = []
for col in heart_disease_df.columns:
    if heart_disease_df[col].dtype == 'str':
        cat_cols.append(col)
print(cat_cols)

# BLIND LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = heart_disease_df.drop('Heart_Disease', axis=1)
y = heart_disease_df['Heart_Disease']

te_cols = ['General_Health', 'Checkup', 'Age_Category']
ohe_cols = [col for col in cat_cols if col not in te_cols and col != 'Heart_Disease']
#print(te_cols, ohe_cols)

# Define the Preprocessing (Encoding Only)
# We don't scale here yet because OHE/Target outputs aren't created until this runs.
preprocessor = ColumnTransformer(
    transformers=[
        #('num', 'passthrough', ['income', 'age']), # Keep numbers as-is for now
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols),
        ('te', TargetEncoder(target_type='binary'), te_cols)
    ])

# Create the Master Pipeline
# The Order: Encode -> Scale EVERY column -> Logistic Regression
full_pipeline = Pipeline(steps=[
    ('encoding', preprocessor),
    ('scaling', StandardScaler()), # This scales the outputs of all encoders + numeric columns
    ('classifier', LogisticRegression())
])

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

full_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = full_pipeline.predict(X_test)
print("--- Classification Report (Scaled Everything) ---")
print(classification_report(y_test, y_pred))


# NOW we will apply PCA and see how it affects

# Build the Master Pipeline with PCA
# Order: Encode -> Scale -> PCA -> Logistic Regression
pca_pipeline = Pipeline(steps=[
    ('encoding', preprocessor),
    ('scaling', StandardScaler()), 
    ('pca', PCA(n_components=0.95)), # Keep components explaining 95% of variance
    ('classifier', LogisticRegression())
])

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The pipeline handles everything: 
pca_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pca_pipeline.predict(X_test)

print("--- PCA Pipeline Classification Report ---")
print(classification_report(y_test, y_pred))

# Deep Dive into PCA results
n_components = pca_pipeline.named_steps['pca'].n_components_
total_variance = np.sum(pca_pipeline.named_steps['pca'].explained_variance_ratio_)

print(f"Original feature count after encoding: ~6-8 (depending on unique cities)")
print(f"PCA reduced this to: {n_components} components")
print(f"Total variance captured by PCA: {total_variance:.2%}")


# NOW we will apply logic to handle imbalanced data
# Build the Master Pipeline with PCA
# Order: Encode -> Scale -> PCA -> Logistic Regression
pca_pipeline = Pipeline(steps=[
    ('encoding', preprocessor),
    ('scaling', StandardScaler()), 
    ('pca', PCA(n_components=0.95)), # Keep components explaining 95% of variance
    ('classifier', LogisticRegression(class_weight='balanced')) # TWEAK
])

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The pipeline handles everything: 
pca_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pca_pipeline.predict(X_test)

print("--- PCA Pipeline Classification Report ---")
print(classification_report(y_test, y_pred))

# Deep Dive into PCA results
n_components = pca_pipeline.named_steps['pca'].n_components_
total_variance = np.sum(pca_pipeline.named_steps['pca'].explained_variance_ratio_)

print(f"Original feature count after encoding: ~6-8 (depending on unique cities)")
print(f"PCA reduced this to: {n_components} components")
print(f"Total variance captured by PCA: {total_variance:.2%}")

# FIRST we will remove data 75% having target class same proportion as original data size as SVM is quite slow for huge data

print("Original Proportions:")
print(heart_disease_df['Heart_Disease'].value_counts(normalize=True))

# Remove 75% of the data proportionally
# test_size=0.5 means we "discard" 50% and "keep" 50%
# stratify=df['target'] ensures the 80/20 split remains the same
heart_disease_kept_df, heart_disease_discarded_df = train_test_split(
    heart_disease_df, 
    test_size=0.75, 
    random_state=42, 
    stratify=heart_disease_df['Heart_Disease']
)

print("\nNew Proportions (after removing 50%):")
print(heart_disease_kept_df['Heart_Disease'].value_counts(normalize=True))
print(f"\nTotal rows remaining: {len(heart_disease_kept_df)}")

heart_disease_kept_df.describe()
heart_disease_kept_df.info()
heart_disease_kept_df.head(5)


# NOW we will work on LDA + SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

# runs on modified dataset
X = heart_disease_kept_df.drop('Heart_Disease', axis=1)
y = heart_disease_kept_df['Heart_Disease']

# Preprocessing (Encode Categories)
# WE will borrow from previous one

# Master Pipeline: Encode -> Scale -> LDA -> SVM
# Note: LDA is supervised, so it needs 'y' during the fit process
lda_svm_pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('scale', StandardScaler()),
    ('lda', LDA(n_components=1)), # Binary class = max 1 component
    ('svm', SVC(kernel='linear', class_weight='balanced'))
])

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# LDA will use y_train to find the best separation axis
lda_svm_pipeline.fit(X_train, y_train)

# Interpret LDA before looking at SVM
lda_step = lda_svm_pipeline.named_steps['lda']
print(f"Variance explained by LDA component: {lda_step.explained_variance_ratio_[0]:.2%}")

# Evaluate Final Performance
y_pred = lda_svm_pipeline.predict(X_test)
print("\n--- SVM + LDA Classification Report ---")
print(classification_report(y_test, y_pred))
