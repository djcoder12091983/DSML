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