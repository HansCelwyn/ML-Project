import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

student_df = pd.read_csv("../../student_lifestyle_100k.csv")

print(student_df.head())
print("count")
print(student_df.count())
print("info")
print(student_df.info())

numerical_columns = ['Age', 'CGPA', 'Sleep_Duration', 'Study_Hours', 'Social_Media_Hours', 'Physical_Activity', 'Stress_Level']
categorical_columns = ['Gender', 'Department']

data_numerical = student_df[numerical_columns]
data_categorical = student_df[categorical_columns]

ohencoder = OneHotEncoder(sparse_output=False)
data_encoded = ohencoder.fit_transform(data_categorical)

print("\nEncoded categories:")
for i, col in enumerate(categorical_columns):
    print(col, ":", ohencoder.categories_[i])

X = np.concatenate([data_numerical.values, data_encoded], axis=1)
y = student_df['Depression'].astype(int)

#Normal experiments
def run_experiment(exp_name, hidden_layer_sizes, test_size=0.30, activation='relu'):
    print(f"\n{'='*20}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  hidden_layer_sizes={hidden_layer_sizes}  |  test_size={test_size}  |  activation={activation}")
    print(f"{'='*20}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size:     {X_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)


    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=100, activation=activation, random_state=45)

    print(f"\n Max_iter = 100")
    print("Model parameters:")
    print(mlp.get_params())

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))


# Experiment 6 Setup
def run_experiment_smote(exp_name, hidden_layer_sizes, test_size=0.30, activation='relu'):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  hidden_layer_sizes={hidden_layer_sizes}  |  test_size={test_size}  |  activation={activation}  |  SMOTE=ON")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)
    print(f"Training set size (before SMOTE): {X_train.shape}")
    print(f"Test set size:                    {X_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # Apply SMOTE only on training data
    print("\nClass distribution BEFORE SMOTE:")
    print(pd.Series(y_train).value_counts())

    smote = SMOTE(random_state=45)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nClass distribution AFTER SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
    print(f"Training set size (after SMOTE): {X_train_resampled.shape}")


    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=100, activation=activation, random_state=45)

    print(f"\n--- max_iter = {i} ---")
    print("Model parameters:")
    print(mlp.get_params())

    mlp.fit(X_train_resampled, y_train_resampled)

    y_pred = mlp.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))


# Experiment 1: Baseline — varying max_iter only
experiments_1 = [
    ("Exp 1 — Baseline (20,20,20) | test=0.3 | relu", (20, 20, 20), 0.30, 'relu'),
]

# Experiment 2: Number of hidden layers
experiments_2 = [
    ("Exp 2a — 1 hidden layer    (20)            | test=0.3 | relu", (20),            0.30, 'relu'),
    ("Exp 2b — 4 hidden layers   (20,20,20,20)    | test=0.3 | relu", (20,20,20,20),    0.30, 'relu'),
]

# Experiment 3: Number of neurons per hidden layer
experiments_3 = [
    ("Exp 3a — (100,100)          | test=0.3 | relu", (100,100),         0.30, 'relu'),
    ("Exp 3b — (500,200,300)      | test=0.3 | relu", (500,200,300),     0.30, 'relu'),
    ("Exp 3c — (100,150,200,250)  | test=0.3 | relu", (100,150,200,250), 0.30, 'relu'),
]

# Experiment 4: Train-test split size
experiments_4 = [
    ("Exp 4a — (20,20,20) | test=0.2 (80/20 split) | relu", (20,20,20), 0.20, 'relu'),
    ("Exp 4b — (20,20,20) | test=0.5 (50/50 split) | relu", (20,20,20), 0.50, 'relu'),
]

# Experiment 5: Activation function (tanh)
experiments_5 = [
    ("Exp 5  — (20,20,20) | test=0.7 | tanh", (20,20,20), 0.50, 'tanh'),
]

# Experiment 6: SMOTE — class imbalance handling
experiments_6 = [
    ("Exp 6a — SMOTE | (20,20,20)    | test=0.3 | relu", (20,20,20),    0.30, 'relu'),
    ("Exp 6b — SMOTE | (20,20,20)    | test=0.3 | tanh", (20,20,20),    0.30, 'tanh'),
    ("Exp 6c — SMOTE | (100,100)     | test=0.3 | relu", (100,100),     0.30, 'relu'),
    ("Exp 6d — SMOTE | (500,200,300) | test=0.3 | relu", (500,200,300), 0.30, 'relu'),
]

# ─── Run All Experiments ─────────────────────────────────────────────────────

# Experiments 2 – 5: standard (no SMOTE)
standard_experiments = (experiments_4 + experiments_5)

for exp_name, hidden, test_size, activation in standard_experiments:
    run_experiment(exp_name, hidden, test_size, activation)

# Experiment 6: SMOTE
for exp_name, hidden, test_size, activation in experiments_6:
    run_experiment_smote(exp_name, hidden, test_size, activation)

print("\n" + "="*20)
