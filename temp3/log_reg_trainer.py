import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class LogisticRegressionTrainer:
    def __init__(self, data, target_column):
        """
        Logistic Regression Trainer Class
        :param data: DataFrame containing features and labels
        :param target_column: Name of the target column in the data
        """
        self.data = data
        self.target_column = target_column

    def preprocess_data(self, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Splits the data into train, validation, and test sets.
        :param test_size: Proportion of the data to be used for testing
        :param validation_size: Proportion of the data to be used for validation
        :param random_state: Random state for reproducibility
        :return: Train, validation, and test sets
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Split train + validation and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Split train and validation
        validation_fraction = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_fraction, random_state=random_state, stratify=y_train
        )

        print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Trains a Logistic Regression model and evaluates it.
        :return: Model, Validation Metrics, Test Metrics
        """
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial", class_weight="balanced")
        model.fit(X_train_scaled, y_train)

        # Validation Performance
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print("\nValidation Performance:")
        print(f"Accuracy: {val_accuracy * 100:.2f}%")
        print(classification_report(y_val, y_val_pred))

        # Test Performance
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("\nTest Performance:")
        print(f"Accuracy: {test_accuracy * 100:.2f}%")
        print(classification_report(y_test, y_test_pred))

        return model, val_accuracy, test_accuracy, y_val_pred, y_test_pred

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
        """
        Plots a confusion matrix.
        :param y_true: Ground truth labels
        :param y_pred: Predicted labels
        :param labels: List of class labels
        :param title: Title of the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


if __name__ == "__main__":
    # Load the extracted feature data
    file_path = "./dataset/emg_features.csv"
    data = pd.read_csv(file_path)

    # Specify the target column
    target_column = "class"

    # Initialize the trainer
    trainer = LogisticRegressionTrainer(data, target_column)

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.preprocess_data()

    # Train and evaluate the model
    model, val_accuracy, test_accuracy, y_val_pred, y_test_pred = trainer.train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Plot confusion matrix for test set
    trainer.plot_confusion_matrix(y_test, y_test_pred, labels=data[target_column].unique(), title="Test Set Confusion Matrix")
