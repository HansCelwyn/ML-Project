import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

student_df = pd.read_csv("../../student_lifestyle_100k.csv")
print(student_df.head())

print("count")
print(student_df.count())
print("info")
print(student_df.info())

sns.heatmap(student_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

print("Missing values")
print(student_df.isnull().sum())

if student_df.isnull().sum().sum() > 0:
    print("Removing rows with missing data")
    student_df.dropna(inplace=True)
    print("count after cleaning")
    print(student_df.count())

numerical_cols = ['Age', 'CGPA', 'Sleep_Duration', 'Study_Hours', 'Social_Media_Hours', 'Physical_Activity', 'Stress_Level']
categorical_cols = ['Gender', 'Department']

X_numerical = student_df[numerical_cols].values

encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(student_df[categorical_cols])

print("Encoded categorical features")
print("Categories:", encoder.categories_)

X = np.concatenate([X_numerical, X_categorical], axis=1)


y = student_df['Depression'].astype(int)

print(f"Feature matrix shape: {X.shape}")

print(f"Target variable distribution: {y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

lrmodel = LogisticRegression(max_iter=1000, class_weight='balanced')
lrmodel.fit(X_train, y_train)

pred_result = lrmodel.predict(X_test)

cm = confusion_matrix(y_test, pred_result)
print("Confusion Matrix")
print(cm)

c_report = classification_report(y_test, pred_result)
print("Classification Report")
print(c_report)

acc = accuracy_score(y_test, pred_result)
print(f"Accuracy: {acc:.4f}")

sns.heatmap(cm, annot=True, fmt='d')

plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
