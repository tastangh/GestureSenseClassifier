import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def evaluate_model(y_true, y_pred, title, save_path):
    """
    Evaluate the model's performance with various metrics and visualize the confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualize and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_true), yticklabels=set(y_true))
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.show()

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

def extract_features(emg_data):
    """
    Extract features for each row in the dataset.
    """
    X = emg_data.iloc[:, :-1].values  # Exclude the class label column
    y = emg_data['class'].values  # Class labels

    # Feature extraction for each row
    mav = np.mean(np.abs(X), axis=1)  # Mean Absolute Value
    rms = np.sqrt(np.mean(np.square(X), axis=1))  # Root Mean Square
    wl = np.sum(np.abs(np.diff(X, axis=1)), axis=1)  # Waveform Length
    zc = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1)  # Zero Crossing
    ssc = np.sum(np.diff(np.sign(np.diff(X, axis=1)), axis=1) != 0, axis=1)  # Slope Sign Change

    # Combine features into a single feature matrix
    features = np.column_stack([mav, rms, wl, zc, ssc])

    return pd.DataFrame(features), y

if __name__ == "__main__":
    # Step 1: Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data("dataset/rawData/0.csv", "dataset/rawData/1.csv", "dataset/rawData/2.csv", "dataset/rawData/3.csv")

    # Step 2: Shuffle the data
    print("Shuffling data...")
    raw_data = shuffle(raw_data, random_state=42)

    # Step 3: Extract features
    print("Extracting features...")
    X, y = extract_features(raw_data)

    # Step 4: Split into training, validation, and test sets
    print("Splitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    # Step 5: Train the Logistic Regression model
    print("Training Logistic Regression model on feature-extracted data...")
    model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
    model.fit(X_train, y_train)

    # Step 6: Validate the model
    print("\nValidating the model...")
    y_val_pred = model.predict(X_val)
    print("Validation Performance:")
    evaluate_model(y_val, y_val_pred, "Validation", "validation_confusion_matrix_features.png")

    # Step 7: Test the model
    print("\nTesting the model...")
    y_test_pred = model.predict(X_test)
    print("Test Performance:")
    evaluate_model(y_test, y_test_pred, "Test", "test_confusion_matrix_features.png")
