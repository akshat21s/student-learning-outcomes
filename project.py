import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, shapiro
import warnings
warnings.filterwarnings("ignore")

file_path = "/Users/akshatjaiswal/Desktop/SEM 4/INT375/Project/7055_source_data.csv"
df = pd.read_csv(file_path)

print("Data Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

learning_cols = df.columns[6:30]
df[learning_cols] = df[learning_cols].apply(lambda x: x.fillna(x.mean()))

df["Average_Score"] = df[learning_cols].mean(axis=1)

df_state_avg = df.groupby("srcStateName")["Average_Score"].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
df_state_avg.plot(kind='bar', color='pink')
plt.title('Average Student Score by State')
plt.ylabel('Average Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



selected_cols = df[learning_cols].iloc[:, :5] 

corr_matrix = selected_cols.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation (First 5 Learning Outcomes)")
plt.tight_layout()
plt.show()

print("\nSummary Stats for Average Score:")
print(df["Average_Score"].describe())

corr_val = df[["Average_Score", "Students surveyed"]].corr()
print("\nCorrelation with Students surveyed:\n", corr_val)

Q1 = df["Average_Score"].quantile(0.25)
Q3 = df["Average_Score"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Average_Score"] < (Q1 - 1.5 * IQR)) | (df["Average_Score"] > (Q3 + 1.5 * IQR))]
print(f"\nDetected {len(outliers)} outliers in Average Score.")

state1 = "Uttar Pradesh"
state2 = "Assam"
scores1 = df[df["srcStateName"] == state1]["Average_Score"]
scores2 = df[df["srcStateName"] == state2]["Average_Score"]

t_stat, p_val = ttest_ind(scores1, scores2, nan_policy='omit')
print(f"\nT-test between {state1} and {state2}:")
print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.4f}")

shapiro_test = shapiro(df["Average_Score"][:500])
print("\nShapiro-Wilk Test for Normality (first 500 entries):")
print("W-statistic:", shapiro_test.statistic, "P-value:", shapiro_test.pvalue)

plt.figure(figsize=(10,6))
sns.histplot(df["Average_Score"], kde=True, color='orange')
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Frequency")
plt.show()

contingency = pd.crosstab(df['srcStateName'], df['Class Studying'])
chi2, p, dof, ex = chi2_contingency(contingency)
print("\nChi-squared test for Class Studying across States:")
print(f"Chi2 Statistic: {chi2:.2f}, P-value: {p:.4f}, DOF: {dof}")

plt.figure(figsize=(8,5))
plt.hist(df["Average_Score"], bins=20, color='teal', edgecolor='black')
plt.title("Histogram of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Students surveyed"], df["Average_Score"], alpha=0.5, color='purple')
plt.title("Average Score vs. Students Surveyed")
plt.xlabel("Students Surveyed")
plt.ylabel("Average Score")
plt.grid(True)
plt.tight_layout()
plt.show()

year_counts = df['Year'].value_counts()

plt.figure(figsize=(7,7))
plt.pie(year_counts, labels=year_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Distribution of Data by Year")
plt.tight_layout()
plt.show()

top_states = df['srcStateName'].value_counts().nlargest(6)

plt.figure(figsize=(8,8))
plt.pie(top_states, labels=top_states.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
plt.title("Top 6 States by Data Entries")
plt.tight_layout()
plt.show()