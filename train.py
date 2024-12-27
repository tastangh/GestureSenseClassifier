import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
from feature_selector import FeatureSelector


# Sabitler ve dosya yolları
RAW_DATA_PATH = "dataset/emg_data.csv"
FILTERED_DATA_PATH = "filtered_dataset/emg_filtered_data.csv"
RESULTS_DIR = "results"
FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save_model(X_train, y_train, output_path, model_params=None):
    """
    Verilen eğitim verileriyle bir model eğitir ve modeli dosyaya kaydeder.
    :param X_train: Eğitim özellik matrisi
    :param y_train: Eğitim hedef değerleri
    :param output_path: Modelin kaydedileceği dosya yolu
    :param model_params: Model parametreleri (opsiyonel)
    """
    print(f"Model {output_path} için eğitiliyor...")
    if model_params is None:
        model_params = {"random_state": 42, "n_estimators": 100}
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # Modelin kaydedilmesi
    pd.to_pickle(model, output_path)
    print(f"Model kaydedildi: {output_path}")
    return model


def evaluate_model(model, X_val, y_val, label):
    """
    Modeli doğrulama veri seti üzerinde değerlendirir.
    :param model: Eğitimli model
    :param X_val: Doğrulama özellik matrisi
    :param y_val: Doğrulama hedef değerleri
    :param label: Değerlendirme raporu için etiket
    """
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"\n{label} - Doğruluk: {accuracy}")
    print(f"{label} - Sınıflandırma Raporu:\n", classification_report(y_val, val_predictions))
    return accuracy


def main():
    # 1. Veri yükleme
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # Ham veri
    raw_processor = DataProcessor(class_names)
    raw_processor.set_data_path(RAW_DATA_PATH)
    raw_dataset = raw_processor.load_data()

    # Filtrelenmiş veri
    filtered_processor = DataProcessor(class_names)
    filtered_processor.set_data_path(FILTERED_DATA_PATH)
    filtered_dataset = filtered_processor.load_data()

    # 2. Özellik çıkarımı
    extractor = FeatureExtractor(window_size=8)

    print("Ham veri için özellikler çıkarılıyor...")
    raw_features_df = extractor.extract_features(raw_dataset)
    raw_features_path = os.path.join(FEATURES_DIR, "raw_features_with_labels.csv")
    raw_features_df.to_csv(raw_features_path, index=False)

    print("Filtrelenmiş veri için özellikler çıkarılıyor...")
    filtered_features_df = extractor.extract_features(filtered_dataset)
    filtered_features_path = os.path.join(FEATURES_DIR, "filtered_features_with_labels.csv")
    filtered_features_df.to_csv(filtered_features_path, index=False)

    # 3. Özellik seçimi ve ağırlıklandırma
    raw_selector = FeatureSelector(raw_features_df, output_dir=os.path.join(RESULTS_DIR, "feature_selection/raw"))
    filtered_selector = FeatureSelector(filtered_features_df, output_dir=os.path.join(RESULTS_DIR, "feature_selection/filtered"))

    print("Ham veri için ağırlıklandırılmış özellikler seçiliyor...")
    raw_weights = raw_selector.compute_feature_weights(method="f_classif")
    raw_features_weighted = raw_features_df[raw_weights["Feature"].values]

    print("Filtrelenmiş veri için ağırlıklandırılmış özellikler seçiliyor...")
    filtered_weights = filtered_selector.compute_feature_weights(method="f_classif")
    filtered_features_weighted = filtered_features_df[filtered_weights["Feature"].values]

    # 4. Eğitim ve doğrulama için veri bölme
    splits = {}
    splits["raw"] = train_test_split(raw_features_df.drop(columns=["Label"]), raw_features_df["Label"], test_size=0.2, random_state=42)
    splits["filtered"] = train_test_split(filtered_features_df.drop(columns=["Label"]), filtered_features_df["Label"], test_size=0.2, random_state=42)
    splits["raw_weighted"] = train_test_split(raw_features_weighted, raw_features_df["Label"], test_size=0.2, random_state=42)
    splits["filtered_weighted"] = train_test_split(filtered_features_weighted, filtered_features_df["Label"], test_size=0.2, random_state=42)

    # 5. Modellerin eğitimi ve değerlendirmesi
    for key, (X_train, X_val, y_train, y_val) in splits.items():
        print(f"\n--- {key.upper()} için eğitim başlıyor ---")
        model_path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
        model = train_and_save_model(X_train, y_train, model_path)
        evaluate_model(model, X_val, y_val, label=f"{key.upper()} Modeli")

    print("\n--- Tüm Eğitim İşlemleri Tamamlandı ---")


if __name__ == "__main__":
    main()
