import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dosya yolları
RAW_DATA_PATH = "dataset/emg_data.csv"
FEATURES_DATA_PATH = "features_emg_data_normalized.csv"
RESULTS_DIR = "results/logistic_regression"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_and_split(data_path, label_column="Gesture_Class", test_size=0.2, validation_size=0.2):
    """
    Veriyi yükler ve eğitim/validation/test olarak böler.
    """
    data = pd.read_csv(data_path)
    X = data.drop(columns=[label_column]).values
    y = pd.get_dummies(data[label_column]).values  # One-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, X_val, y_train, y_val, input_dim, epochs=100, batch_size=32):
    """
    Logistic regression modelini eğitir.
    """
    model = Sequential([
        Dense(y_train.shape[1], input_dim=input_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Modeli test setinde değerlendirir.
    """
    y_pred = model.predict(X_test)
    test_loss = log_loss(y_test, y_pred)
    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    return test_loss, test_accuracy, report

def main():
    results = {}
    
    # 1. Ham Veri
    print("\n--- Logistic Regression: Ham Veri ---")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_and_split(RAW_DATA_PATH)
    model, history = train_logistic_regression(X_train, X_val, y_train, y_val, input_dim=X_train.shape[1])
    loss, accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Ham Veri - Test Loss: {loss}, Test Accuracy: {accuracy}\n{report}")
    results["Ham Veri"] = {"Loss": loss, "Accuracy": accuracy}
    
    # 2. Normalized Özellik Çıkarımı
    print("\n--- Logistic Regression: Özellik Çıkarımı (Normalized) ---")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_and_split(FEATURES_DATA_PATH)
    model, history = train_logistic_regression(X_train, X_val, y_train, y_val, input_dim=X_train.shape[1])
    loss, accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Özellik Çıkarımı - Test Loss: {loss}, Test Accuracy: {accuracy}\n{report}")
    results["Normalized Özellik Çıkarımı"] = {"Loss": loss, "Accuracy": accuracy}
    
    # Sonuçları Kaydetme
    results_path = os.path.join(RESULTS_DIR, "logistic_regression_results.json")
    pd.DataFrame(results).to_json(results_path, orient="index")
    print(f"\nSonuçlar kaydedildi: {results_path}")

if __name__ == "__main__":
    main()
