from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

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

if __name__ == "__main__":
    # Step 1: Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data("dataset/rawData/0.csv", "dataset/rawData/1.csv", "dataset/rawData/2.csv", "dataset/rawData/3.csv")

    # Step 2: Shuffle the data
    print("Shuffling data...")
    raw_data = shuffle(raw_data, random_state=42)

    # Step 3: Split features and labels
    X = raw_data.iloc[:, :-1].values  # Features (all except the last column)
    y = raw_data['class'].values     # Labels (last column)

    # Step 4: Split into training, validation, and test sets
    print("Splitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    # Step 5: Train the Logistic Regression model
    print("Training Logistic Regression model on shuffled data...")
    model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
    model.fit(X_train, y_train)

    # Step 6: Validate the model
    print("\nValidating the model...")
    y_val_pred = model.predict(X_val)
    print("Validation Performance:")
    evaluate_model(y_val, y_val_pred)

    # Step 7: Test the model
    print("\nTesting the model...")
    y_test_pred = model.predict(X_test)
    print("Test Performance:")
    evaluate_model(y_test, y_test_pred)
