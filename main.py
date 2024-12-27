import os
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import shap


def load_and_extract_features(data_path, class_names, output_dir):
    """
    Ham veriyi işler ve özellikleri çıkarır.
    """
    # Ham veriyi yükle
    processor = DataProcessor(data_path, class_names)
    dataset = processor.load_data()

    # Özellikleri ve etiketleri ayır
    X_raw = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Özellik çıkarımı
    extractor = FeatureExtractor(window_size=8)
    features_df = extractor.extract_from_data(X_raw, y)

    # Çıkarılan özellikleri kaydet
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "feature_matrix_with_labels.csv")
    features_df.to_csv(output_file, index=False)
    print(f"Özellik matrisi '{output_file}' dosyasına kaydedildi.")

    return features_df


def split_data(features_df, validation_size=0.2, test_size=0.2):
    """
    Veriyi eğitim, doğrulama ve test setlerine ayırır.
    """
    X = features_df.drop(columns=["Label"]).values
    y = features_df["Label"].values

    # Önce test setini ayır
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Eğitim ve doğrulama setlerini ayır
    validation_split = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=validation_split,
                                                          random_state=42)

    print(f"Eğitim Seti Boyutu: {X_train.shape}")
    print(f"Doğrulama Seti Boyutu: {X_valid.shape}")
    print(f"Test Seti Boyutu: {X_test.shape}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def feature_importance_analysis(X_train, y_train, X_valid, y_valid, features_df):
    """
    Random Forest ile özellik önemini analiz eder ve görselleştirir.
    """
    # Random Forest ile model eğitimi
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Performans değerlendirme (Doğrulama Seti)
    y_pred_valid = rf_model.predict(X_valid)
    print("\n--- Random Forest Performansı (Validation Set) ---")
    print(classification_report(y_valid, y_pred_valid))
    print(f"Doğruluk: {accuracy_score(y_valid, y_pred_valid):.4f}")

    # Özellik önemini görselleştir
    feature_importances = rf_model.feature_importances_
    feature_names = features_df.columns[:-1]
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx])
    plt.xticks(range(len(sorted_idx)), feature_names[sorted_idx], rotation=90)
    plt.title("Random Forest Özellik Önemi")
    plt.tight_layout()
    plt.show()

    return rf_model


def rfe_feature_selection(X_train, y_train, features_df):
    """
    Recursive Feature Elimination (RFE) ile özellik seçimi.
    """
    svc = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=svc, n_features_to_select=10)
    rfe.fit(X_train, y_train)

    # Seçilen özellikler
    selected_features = features_df.columns[:-1][rfe.support_]
    print("\n--- RFE Seçilen Özellikler ---")
    print(selected_features)

    # Özellik sıralaması
    rfe_ranking = rfe.ranking_
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(rfe_ranking)), rfe_ranking)
    plt.xticks(range(len(rfe_ranking)), features_df.columns[:-1], rotation=90)
    plt.title("RFE Özellik Sıralaması")
    plt.tight_layout()
    plt.show()

    return selected_features, rfe.support_


def shap_analysis(model, X_train, feature_names):
    """
    SHAP analizi ile model açıklanabilirliği.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # SHAP özet grafiği
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)


def evaluate_selected_features(X_train, X_valid, X_test, y_train, y_valid, y_test, selected_features, features_df):
    """
    Seçilen özelliklerle modeli yeniden eğit ve değerlendir.
    """
    # Seçilen özelliklerin isimlerine göre filtrele
    selected_feature_names = list(selected_features)
    X_train_selected = pd.DataFrame(X_train, columns=features_df.columns[:-1])[selected_feature_names].values
    X_valid_selected = pd.DataFrame(X_valid, columns=features_df.columns[:-1])[selected_feature_names].values
    X_test_selected = pd.DataFrame(X_test, columns=features_df.columns[:-1])[selected_feature_names].values

    # Modeli eğit ve değerlendir
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_selected, y_train)

    # Performans değerlendirme (Validation)
    y_pred_valid = model.predict(X_valid_selected)
    print("\n--- Validation Performansı (Seçilen Özellikler) ---")
    print(classification_report(y_valid, y_pred_valid))

    # Performans değerlendirme (Test)
    y_pred_test = model.predict(X_test_selected)
    print("\n--- Test Performansı (Seçilen Özellikler) ---")
    print(classification_report(y_test, y_pred_test))


def main():
    # Dosya yolları ve sınıf isimleri
    raw_data_path = "dataset/emg_data.csv"
    output_dir = "results/features"
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # 1. Veriyi yükle ve özellikleri çıkar
    features_df = load_and_extract_features(raw_data_path, class_names, output_dir)

    # 2. Veriyi Eğitim, Doğrulama ve Test setlerine ayır
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(features_df)

    # 3. Özellik önemini analiz et
    rf_model = feature_importance_analysis(X_train, y_train, X_valid, y_valid, features_df)

    # 4. RFE ile özellik seçimi
    selected_features, selected_support = rfe_feature_selection(X_train, y_train, features_df)

    # 5. SHAP ile model açıklanabilirliği
    shap_analysis(rf_model, X_train, features_df.columns[:-1])

    # 6. Seçilen özelliklerle performans analizi
    evaluate_selected_features(X_train, X_valid, X_test, y_train, y_valid, y_test, selected_features, features_df)

    print("\n--- Tüm İşlemler Tamamlandı ---")


if __name__ == "__main__":
    main()
