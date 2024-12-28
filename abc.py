import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter

def load_raw_data(*files):
    """
    Load multiple CSV files and combine them with class labels for raw data processing.
    """
    dataframes = []
    for idx, file in enumerate(files):
        df = pd.read_csv(file, header=None)
        df['class'] = idx  # Add class label based on file index
        dataframes.append(df)
    combined_data = pd.concat(dataframes, axis=0).reset_index(drop=True)
    return combined_data

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance with metrics and display confusion matrix.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def visualize_feature_distribution(X, y):
    """
    Visualize the distribution of features for each class.
    """
    for i in range(X.shape[1]):
        plt.figure(figsize=(8, 4))
        for label in np.unique(y):
            plt.hist(X[y == label, i], bins=30, alpha=0.5, label=f'Class {label}')
        plt.title(f'Feature {i} Distribution')
        plt.legend()
        plt.show()

def check_feature_correlation(X, y):
    """
    Check feature correlation with labels.
    """
    df = pd.DataFrame(X)
    df['label'] = y
    correlations = df.corr()['label'].sort_values(ascending=False)
    print("Feature Correlations with Labels:")
    print(correlations)

def test_with_noise(X, y):
    """
    Test model performance with added noise to the data.
    """
    noise = np.random.normal(0, 0.1, X.shape)
    X_noisy = X + noise
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    print("Performance with Noise:")
    evaluate_model(y_test, y_test_pred)

def analyze_class_distribution(y_train, y_val, y_test):
    """
    Analyze the class distribution in train, validation, and test sets.
    """
    print("Class Distribution in Training Set:", Counter(y_train))
    print("Class Distribution in Validation Set:", Counter(y_val))
    print("Class Distribution in Test Set:", Counter(y_test))

if __name__ == "__main__":
    # Step 1: Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data("dataset/rawData/0.csv", "dataset/rawData/1.csv", "dataset/rawData/2.csv", "dataset/rawData/3.csv")

    # Step 2: Shuffle the data
    print("Shuffling data...")
    raw_data = shuffle(raw_data, random_state=42)

    # Step 3: Split features and labels
    X = raw_data.iloc[:, :-1].values  # Features
    y = raw_data['class'].values     # Labels

    # Step 4: Split into training, validation, and test sets
    print("Splitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Step 5: Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Validate the model
    print("\nValidating the model...")
    y_val_pred = model.predict(X_val)
    evaluate_model(y_val, y_val_pred)

    # Step 7: Test the model
    print("\nTesting the model...")
    y_test_pred = model.predict(X_test)
    evaluate_model(y_test, y_test_pred)

    # Step 8: Analyze Class Distribution
    print("\nAnalyzing class distribution...")
    analyze_class_distribution(y_train, y_val, y_test)

    # Step 9: Visualize Feature Distribution
    print("\nVisualizing feature distributions...")
    visualize_feature_distribution(X, y)

    # Step 10: Check Feature Correlation
    print("\nChecking feature correlations...")
    check_feature_correlation(X, y)

    # Step 11: Test with Added Noise
    print("\nTesting with added noise...")
    test_with_noise(X, y)
