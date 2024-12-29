import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

class EMGDataProcessor:
    def __init__(self):
        self.data = None
        self.scaler = None

    def load_and_combine(self, *files):
        """
        Load multiple CSV files and combine them with class labels.
        """
        dataframes = []
        for idx, file in enumerate(files):
            df = pd.read_csv(file, header=None)
            df['class'] = idx  # Add class label based on file index
            dataframes.append(df)
        self.data = pd.concat(dataframes, axis=0).reset_index(drop=True)
        return self.data

    def clean_outliers(self, method="clip", z_thresh=3):
        """
        Handle outliers in the dataset.
        method: 'clip' or 'remove'.
        z_thresh: Z-score threshold for identifying outliers.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use load_and_combine first.")
        
        features = self.data.iloc[:, :-1]  # Exclude class column
        if method == "clip":
            # Clip values beyond 3 standard deviations
            mean = features.mean()
            std = features.std()
            lower_bound = mean - z_thresh * std
            upper_bound = mean + z_thresh * std
            self.data.iloc[:, :-1] = features.clip(lower=lower_bound, upper=upper_bound, axis=1)
        elif method == "remove":
            # Remove rows with outliers
            z_scores = ((features - features.mean()) / features.std()).abs()
            self.data = self.data[(z_scores < z_thresh).all(axis=1)]
        return self.data

    def scale_data(self, method="minmax"):
        """
        Scale the data using MinMaxScaler or StandardScaler.
        method: 'minmax' or 'standard'.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use load_and_combine first.")
        
        features = self.data.iloc[:, :-1]  # Exclude class column
        if method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        self.data.iloc[:, :-1] = self.scaler.fit_transform(features)
        return self.data

    def create_sliding_windows(self, window_size=5, step_size=1):
        """
        Create sliding windows for the data.
        window_size: Number of rows per window.
        step_size: Step size between windows.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use load_and_combine first.")
        
        X, y = [], []
        features = self.data.iloc[:, :-1].values
        labels = self.data.iloc[:, -1].values

        for i in range(0, len(features) - window_size + 1, step_size):
            window = features[i:i + window_size].flatten()
            X.append(window)
            y.append(labels[i])  # Use the label of the first row in the window

        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into training, validation, and test sets.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def balance_data(self, X, y, method="oversample"):
        """
        Balance the dataset using oversampling or undersampling.
        method: 'oversample' or 'undersample'.
        """
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler

        if method == "oversample":
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
        elif method == "undersample":
            undersampler = RandomUnderSampler(random_state=42)
            X_res, y_res = undersampler.fit_resample(X, y)
        return X_res, y_res

    def save_to_csv(self, X, y, filename):
        """
        Save processed data to a CSV file.
        """
        data_to_save = pd.DataFrame(X)
        data_to_save['label'] = y
        data_to_save.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Initialize the processor
    processor = EMGDataProcessor()

    # Step 1: Load and combine datasets
    processor.load_and_combine("dataset/rawData/0.csv", "dataset/rawData/1.csv", "dataset/rawData/2.csv", "dataset/rawData/3.csv")

    # Step 2: Clean outliers
    processor.clean_outliers(method="clip")

    # Step 3: Scale the data
    processor.scale_data(method="standard")

    # Step 4: Create sliding windows
    X, y = processor.create_sliding_windows(window_size=5, step_size=1)

    # Step 5: Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

    # Step 6: Balance the training data
    X_train_balanced, y_train_balanced = processor.balance_data(X_train, y_train, method="oversample")

    # Step 7: Display class distributions
    print("Class distribution in training data (balanced):", Counter(y_train_balanced))
    print("Class distribution in validation data:", Counter(y_val))
    print("Class distribution in test data:", Counter(y_test))

    # Step 8: Save the processed datasets
    processor.save_to_csv(X_train_balanced, y_train_balanced, "train_balanced.csv")
    processor.save_to_csv(X_val, y_val, "validation.csv")
    processor.save_to_csv(X_test, y_test, "test.csv")
