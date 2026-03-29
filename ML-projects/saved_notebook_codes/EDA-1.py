import pandas as pd
jobsalary_df = pd.read_csv('../data/job_salary_prediction_dataset.csv')

jobsalary_df.describe()
jobsalary_df.info()
jobsalary_df.head(5)

import matplotlib.pyplot as plt

# Setup the side-by-side plotting area
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plotting the Histograms
ax1.hist(jobsalary_df['experience_years'], bins=3, color='skyblue', edgecolor='black')
ax1.set_title('Experiences - in years')

ax2.hist(jobsalary_df['skills_count'], bins=3, color='skyblue', edgecolor='black')
ax2.set_title('Skills')

ax3.hist(jobsalary_df['salary'], bins=25, color='skyblue', edgecolor='black')
ax3.set_title('Salary')

plt.tight_layout()
plt.show()

jobsalary_df['job_title'].value_counts().plot(kind='bar', figsize=(8, 5))
jobsalary_df['industry'].value_counts().plot(kind='bar', figsize=(8, 5))
jobsalary_df['location'].value_counts().plot(kind='bar', figsize=(8, 5))

fig, ax = plt.subplots(figsize=(7, 5))

# 'vert=False' makes it horizontal, 'patch_artist=True' allows coloring
ax.boxplot(jobsalary_df['salary'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))

ax.set_ylabel('Salary')
plt.show()


# airlines data
airlines_df = pd.read_csv('../data/airlines_flights_data.csv')

airlines_df.describe()
airlines_df.info()
airlines_df.head(5)

# airlines data
amazon_product_reviews_df = pd.read_csv('../data/Amazon Product Reviews.csv')

amazon_product_reviews_df.describe()
amazon_product_reviews_df.info()
amazon_product_reviews_df.head(5)

# card data
card_df = pd.read_csv('../data/new datasset + classifications/cards_data.csv')

card_df.describe()
card_df.info()
card_df.head(5)

# card transactions data
card_txn_df = pd.read_csv('../data/new datasset + classifications/credit_card_transactions.csv')

card_txn_df.describe()
card_txn_df.info()
card_txn_df.head(5)

# transactions data
txn_df = pd.read_csv('../data/new datasset + classifications/transactions.csv')

txn_df.describe()
txn_df.info()
txn_df.head(5)

# tweet engagements data
tweets_df = pd.read_csv('../data/new datasset + classifications/tweets-engagement-metrics.csv')

tweets_df.describe()
tweets_df.info()
tweets_df.head(5)

# tweet engagements data
amazon_sales_df = pd.read_csv('../data/Amazon Sales Dataset.csv')

amazon_sales_df.describe()
amazon_sales_df.info()
amazon_sales_df.head(5)

# tweet engagements data
cinema_ticket_df = pd.read_csv('../data/cinemaTicket_Ref.csv')

cinema_ticket_df.describe()
cinema_ticket_df.info()
cinema_ticket_df.head(5)

# netflix data
netflix_titles_df = pd.read_csv('../data/netflix_titles_2021.csv')

netflix_titles_df.describe()
netflix_titles_df.info()
netflix_titles_df.head(5)

# ecommerce dataset
ecommerce_sales_df = pd.read_csv('../data/ecommerce_dataset.csv')

ecommerce_sales_df.describe()
ecommerce_sales_df.info()
ecommerce_sales_df.head(5)

print(detailed_report = ecommerce_sales_df.groupby('CID').count)


import matplotlib.pyplot as plt
import seaborn as sns

# visualization
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
ax = axes.flatten()

sns.countplot(data=ecommerce_sales_df, ax=axes[0], x='Location', hue='Gender')
ax[0].set_title('Location Distribution')

sns.countplot(data=ecommerce_sales_df, ax=axes[1], x='Age Group', hue='Gender')
ax[1].set_title('Age Distribution')

sns.countplot(data=ecommerce_sales_df, ax=axes[2], x='Product Category', hue='Gender')
ax[2].set_title('Product category Distribution')

plt.tight_layout()
plt.show()

# another visualization by sorted values
# List the columns you want to plot
cols_to_plot = ['Location', 'Age Group', 'Product Category', 'Purchase Method']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes = axes.flatten()

# draw each plot
for i, col in enumerate(cols_to_plot):
    sns.countplot(data=ecommerce_sales_df, x=col, ax=axes[i], hue='Gender', 
                  order=ecommerce_sales_df[col].value_counts().index) # Sorting by frequency
    
    # Customizing each subplot
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45) # Rotate labels for readability

plt.tight_layout() # Fixes overlapping titles/labels
plt.show()

# KDE plots for numeric values
# List the columns you want to plot
cols_to_plot = ['Discount Amount (INR)', 'Gross Amount', 'Net Amount']

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
axes = axes.flatten()

# draw each plot
for i, col in enumerate(cols_to_plot):
    sns.kdeplot(data=ecommerce_sales_df, x=col, ax=axes[i], hue='Gender', fill=True)
    
    # Customizing each subplot
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')

plt.tight_layout() # Fixes overlapping titles/labels
plt.show()

# KDE plots for job salary
cols_to_plot = ['job_title', 'education_level', 'industry', 'location']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 15))
axes = axes.flatten()

# draw each plot
for i, col in enumerate(cols_to_plot):
    sns.kdeplot(data=jobsalary_df, x='salary', ax=axes[i], hue=col, fill=True)
    
    # Customizing each subplot
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')

plt.tight_layout() # Fixes overlapping titles/labels
plt.show()