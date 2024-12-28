from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Önceden oluşturduğumuz EMGDataProcessor sınıfını içe aktarın
from emg_data_processor import EMGDataProcessor

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance with various metrics.
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

if __name__ == "__main__":
    # Initialize the processor
    processor = EMGDataProcessor()

    # Step 1: Load and combine datasets
    print("Loading and combining datasets...")
    processor.load_and_combine("dataset/rawData/0.csv", "dataset/rawData/1.csv", "dataset/rawData/2.csv", "dataset/rawData/3.csv")

    # Step 2: Clean outliers
    print("Cleaning outliers...")
    processor.clean_outliers(method="clip")

    # Step 3: Scale the data
    print("Scaling data...")
    processor.scale_data(method="standard")

    # Step 4: Create sliding windows
    print("Creating sliding windows...")
    X, y = processor.create_sliding_windows(window_size=5, step_size=1)

    # Step 5: Split the data
    print("Splitting data into training, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

    # Step 6: Balance the training data
    print("Balancing training data...")
    X_train_balanced, y_train_balanced = processor.balance_data(X_train, y_train, method="oversample")

    # Step 7: Train the Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    # Step 8: Validate the model
    print("\nValidating the model...")
    y_val_pred = model.predict(X_val)
    print("Validation Performance:")
    evaluate_model(y_val, y_val_pred)

    # Step 9: Test the model
    print("\nTesting the model...")
    y_test_pred = model.predict(X_test)
    print("Test Performance:")
    evaluate_model(y_test, y_test_pred)
