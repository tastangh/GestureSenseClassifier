import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# Sabitler ve dosya yolları
RAW_FEATURES_PATH = "results/features/raw_features_with_labels.csv"
FILTERED_FEATURES_PATH = "results/features/filtered_features_with_labels.csv"
RAW_DATA_PATH = "dataset/emg_data.csv"
FILTERED_DATA_PATH = "filtered_dataset/emg_filtered_data.csv"
MODELS_DIR = "results/simple_models"

def load_model(model_path):
    """
    Eğitimli modeli dosyadan yükler.
    """
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print(f"Model yüklendi: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

def evaluate_on_test_data(model, X_test, y_test, label):
    """
    Test seti üzerinde modeli değerlendirir.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{label} - Test Doğruluk: {accuracy}")
    print(f"{label} - Test Sınıflandırma Raporu:\n{classification_report(y_test, y_pred)}")
    return accuracy

def main():
    print("\n--- Test Setinde 4 Senaryonun Karşılaştırması Başlıyor ---")
    
    # Modellerin yolları
    raw_model_path = os.path.join(MODELS_DIR, "Ham Veri_model.pkl")
    filtered_model_path = os.path.join(MODELS_DIR, "Filtrelenmiş Veri_model.pkl")
    raw_features_model_path = os.path.join(MODELS_DIR, "Ham Özellik Çıkarımı_model.pkl")
    filtered_features_model_path = os.path.join(MODELS_DIR, "Filtrelenmiş Özellik Çıkarımı_model.pkl")
    
    # Test setini oluşturma
    print("\n--- Veriler Yükleniyor ---")
    raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
        pd.read_csv(RAW_DATA_PATH).drop(columns=["Gesture_Class"]),
        pd.read_csv(RAW_DATA_PATH)["Gesture_Class"],
        test_size=0.2,
        random_state=42
    )
    
    filtered_X_train, filtered_X_test, filtered_y_train, filtered_y_test = train_test_split(
        pd.read_csv(FILTERED_DATA_PATH).drop(columns=["Gesture_Class"]),
        pd.read_csv(FILTERED_DATA_PATH)["Gesture_Class"],
        test_size=0.2,
        random_state=42
    )
    
    raw_features_df = pd.read_csv(RAW_FEATURES_PATH)
    raw_features_X_train, raw_features_X_test, raw_features_y_train, raw_features_y_test = train_test_split(
        raw_features_df.drop(columns=["Label"]),
        raw_features_df["Label"],
        test_size=0.2,
        random_state=42
    )
    
    filtered_features_df = pd.read_csv(FILTERED_FEATURES_PATH)
    filtered_features_X_train, filtered_features_X_test, filtered_features_y_train, filtered_features_y_test = train_test_split(
        filtered_features_df.drop(columns=["Label"]),
        filtered_features_df["Label"],
        test_size=0.2,
        random_state=42
    )
    
    # Modelleri yükleme
    print("\n--- Modeller Yükleniyor ---")
    raw_model = load_model(raw_model_path)
    filtered_model = load_model(filtered_model_path)
    raw_features_model = load_model(raw_features_model_path)
    filtered_features_model = load_model(filtered_features_model_path)
    
    # Test setinde değerlendirme
    print("\n--- Test Setinde Değerlendirme ---")
    evaluate_on_test_data(raw_model, raw_X_test, raw_y_test, label="Ham Veri Modeli")
    evaluate_on_test_data(filtered_model, filtered_X_test, filtered_y_test, label="Filtrelenmiş Veri Modeli")
    evaluate_on_test_data(raw_features_model, raw_features_X_test, raw_features_y_test, label="Ham Özellik Çıkarımı Modeli")
    evaluate_on_test_data(filtered_features_model, filtered_features_X_test, filtered_features_y_test, label="Filtrelenmiş Özellik Çıkarımı Modeli")
    
    print("\n--- Test Değerlendirmesi Tamamlandı ---")

if __name__ == "__main__":
    main()
