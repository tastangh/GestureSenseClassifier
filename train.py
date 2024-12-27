import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Sabitler ve dosya yolları
RAW_FEATURES_PATH = "results/features/raw_features_with_labels.csv"
FILTERED_FEATURES_PATH = "results/features/filtered_features_with_labels.csv"
RAW_DATA_PATH = "dataset/emg_data.csv"
FILTERED_DATA_PATH = "filtered_emg_data_frequency_domain.csv"
RESULTS_DIR = "results/simple_models"

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_simple_model(X_train, X_val, y_train, y_val, model_name):
    """
    Logistic Regression ile modeli eğitir ve değerlendirir.
    """
    print(f"\n--- {model_name} modeli eğitiliyor ---")
    
    # Basit model tanımı (Logistic Regression)
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    
    # Model değerlendirme
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Modeli Doğruluk: {accuracy}")
    print(f"{model_name} Modeli Sınıflandırma Raporu:\n{classification_report(y_val, y_pred)}")
    
    return model, accuracy

def load_data_and_split(data_path, label_column="Label", test_size=0.2):
    """
    Veriyi yükler ve eğitim/test olarak böler.
    """
    data = pd.read_csv(data_path)
    X = data.drop(columns=[label_column])
    y = data[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def main():
    print("\n--- 4 Senaryonun Karşılaştırması Başlıyor ---")
    
    # 1. Senaryo: Ham Veri
    print("\n[1/4] Ham Veri ile Model Eğitimi ve Değerlendirme")
    raw_X_train, raw_X_val, raw_y_train, raw_y_val = load_data_and_split(RAW_DATA_PATH, label_column="Gesture_Class")
    raw_model, raw_accuracy = train_and_evaluate_simple_model(raw_X_train, raw_X_val, raw_y_train, raw_y_val, "Ham Veri")

    # 2. Senaryo: Filtrelenmiş Veri
    print("\n[2/4] Filtrelenmiş Veri ile Model Eğitimi ve Değerlendirme")
    filtered_X_train, filtered_X_val, filtered_y_train, filtered_y_val = load_data_and_split(FILTERED_DATA_PATH, label_column="Gesture_Class")
    filtered_model, filtered_accuracy = train_and_evaluate_simple_model(filtered_X_train, filtered_X_val, filtered_y_train, filtered_y_val, "Filtrelenmiş Veri")

    # 3. Senaryo: Ham Özellik Çıkarımı
    print("\n[3/4] Ham Özellik Çıkarımı ile Model Eğitimi ve Değerlendirme")
    raw_features_X_train, raw_features_X_val, raw_features_y_train, raw_features_y_val = load_data_and_split(RAW_FEATURES_PATH)
    raw_features_model, raw_features_accuracy = train_and_evaluate_simple_model(raw_features_X_train, raw_features_X_val, raw_features_y_train, raw_features_y_val, "Ham Özellik Çıkarımı")

    # 4. Senaryo: Filtrelenmiş Özellik Çıkarımı
    print("\n[4/4] Filtrelenmiş Özellik Çıkarımı ile Model Eğitimi ve Değerlendirme")
    filtered_features_X_train, filtered_features_X_val, filtered_features_y_train, filtered_features_y_val = load_data_and_split(FILTERED_FEATURES_PATH)
    filtered_features_model, filtered_features_accuracy = train_and_evaluate_simple_model(filtered_features_X_train, filtered_features_X_val, filtered_features_y_train, filtered_features_y_val, "Filtrelenmiş Özellik Çıkarımı")
    
    # Sonuçların Özeti
    print("\n--- SONUÇLAR ---")
    print(f"Ham Veri Modeli Doğruluk: {raw_accuracy}")
    print(f"Filtrelenmiş Veri Modeli Doğruluk: {filtered_accuracy}")
    print(f"Ham Özellik Çıkarımı Modeli Doğruluk: {raw_features_accuracy}")
    print(f"Filtrelenmiş Özellik Çıkarımı Modeli Doğruluk: {filtered_features_accuracy}")
    print("\n--- Karşılaştırma Tamamlandı ---")

if __name__ == "__main__":
    main()
