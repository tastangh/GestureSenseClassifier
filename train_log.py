import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Dosya yolları
RAW_DATA_PATH = "dataset/emg_data.csv"
FEATURES_DATA_PATH = "features_emg_data.csv"
RESULTS_DIR = "results/simple_models"

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_simple_model(X_train, X_val, y_train, y_val, model_name):
    """
    Logistic Regression ile modeli eğitir ve değerlendirir.
    """
    print(f"\n--- {model_name} modeli eğitiliyor ---")
    
    # Logistic Regression modeli
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Modeli Doğruluk: {accuracy}")
    print(f"{model_name} Modeli Sınıflandırma Raporu:\n{classification_report(y_val, y_pred)}")
    
    return model, accuracy

def load_data_and_split(data_path, label_column="Gesture_Class", test_size=0.2):
    """
    Veriyi yükler ve eğitim/test olarak böler.
    """
    data = pd.read_csv(data_path)
    X = data.drop(columns=[label_column])
    y = data[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def main():
    print("\n--- Ham Veri ve Özellik Çıkarımı Karşılaştırması Başlıyor ---")
    
    # 1. Ham Veri
    print("\n[1/2] Ham Veri ile Model Eğitimi ve Değerlendirme")
    raw_X_train, raw_X_val, raw_y_train, raw_y_val = load_data_and_split(RAW_DATA_PATH, label_column="Gesture_Class")
    raw_model, raw_accuracy = train_and_evaluate_simple_model(raw_X_train, raw_X_val, raw_y_train, raw_y_val, "Ham Veri")

    # 2. Özellik Çıkarılmış Veri
    print("\n[2/2] Özellik Çıkarılmış Veri ile Model Eğitimi ve Değerlendirme")
    features_X_train, features_X_val, features_y_train, features_y_val = load_data_and_split(FEATURES_DATA_PATH, label_column="Gesture_Class")
    features_model, features_accuracy = train_and_evaluate_simple_model(features_X_train, features_X_val, features_y_train, features_y_val, "Özellik Çıkarımı")

    # Sonuçların Özeti
    print("\n--- SONUÇLAR ---")
    print(f"Ham Veri Modeli Doğruluk: {raw_accuracy}")
    print(f"Özellik Çıkarımı Modeli Doğruluk: {features_accuracy}")
    print("\n--- Karşılaştırma Tamamlandı ---")

if __name__ == "__main__":
    main()
