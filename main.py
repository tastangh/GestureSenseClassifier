from dataset import DataProcessor
from evaluation import ModelEvaluator
from random_forest_model import RandomForestClassifierModel
from lstm_model import LSTMModel
import os
import numpy as np

# Sınıf isimleri
classes = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

# Veriyi işleme
data_path = "dataset/"
processor = DataProcessor(data_path=data_path, class_names=classes)
dataset = processor.load_data()
X, y = processor.preprocess()
X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

# Eğitim ve test seti dağılım kontrolü
processor.check_train_test_distribution(y_train, y_test)

# Random Forest Modeli
def train_and_evaluate_random_forest():
    print("\n--- Random Forest Modeli Eğitimi ve Değerlendirilmesi ---\n")

    # Random Forest için 2D veri formatına dönüştürme
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (örnek sayısı, 64 özellik)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)    # (örnek sayısı, 64 özellik)

    rf_model = RandomForestClassifierModel(n_estimators=100)
    y_pred_rf = rf_model.train_and_predict(X_train_flat, y_train, X_test_flat)

    # Değerlendirme
    evaluator = ModelEvaluator(class_names=classes)
    evaluator.evaluate(y_test, y_pred_rf, model_name="Random Forest")

# LSTM Modeli
def train_and_evaluate_lstm(saved_model=False):
    print("\n--- LSTM Modeli Eğitimi ve Değerlendirilmesi ---\n")

    # Etiketleri One-Hot Encoding'e çevirme
    y_train_onehot = np.eye(len(classes))[y_train]
    y_test_onehot = np.eye(len(classes))[y_test]

    lstm_model = LSTMModel(n_steps=8, n_features=8, saved_model=saved_model)

    # Model eğitimi
    if not saved_model:
        lstm_model.train(X_train, y_train_onehot, epochs=250, batch_size=32)

    # Tahminler
    y_pred_lstm = lstm_model.predict(X_test)

    # Gerçek ve tahmin edilen sınıfları One-Hot'tan çıkartma
    y_test_classes = np.argmax(y_test_onehot, axis=1)

    # Değerlendirme
    evaluator = ModelEvaluator(class_names=classes)
    evaluator.evaluate(y_test_classes, y_pred_lstm, model_name="LSTM")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Random Forest Modeli
    train_and_evaluate_random_forest()

    # LSTM Modeli
    train_and_evaluate_lstm(saved_model=False)

    print("\n--- Tüm İşlemler Tamamlandı ---")
