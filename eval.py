import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    """
    Farklı veri tiplerini karşılaştırmak için sınıf.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)

    def preprocess_data(self, dataset, feature_only=True):
        """
        Veriyi bağımsız ve bağımlı değişkenlere ayırır.
        :param dataset: DataFrame (etiketler son sütunda yer almalı)
        :param feature_only: Özellik seti mi, ham veri mi kullanılacak
        :return: X (özellikler), y (etiketler)
        """
        if feature_only:
            X = dataset.iloc[:, :-1].values  # Son sütun hariç
            y = dataset.iloc[:, -1].values  # Son sütun etiketler
        else:
            X = dataset.iloc[:, :-1].values  # Ham veri özellikleri
            y = dataset.iloc[:, -1].values
        return X, y

    def evaluate(self, X, y, test_size=0.2):
        """
        Veriyi eğitir ve değerlendirir.
        :param X: Özellikler
        :param y: Etiketler
        :param test_size: Test set oranı
        :return: Değerlendirme metrikleri
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        # Modeli eğit
        self.model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = self.model.predict(X_test)

        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return accuracy, f1, report, cm

    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix"):
        """
        Karışıklık matrisi çizer.
        :param cm: Karışıklık matrisi
        :param class_names: Sınıf isimleri
        :param title: Grafik başlığı
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

if __name__ == "__main__":
    from data_processor import DataProcessor
    from feature_extractor import FeatureExtractor

    # Veri Yolları
    raw_data_path = "dataset/emg_data.csv"
    filtered_data_path = "filtered_dataset/emg_filtered_data.csv"

    # DataProcessor Nesnelerini Kullanarak Verileri Yükle
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    raw_processor = DataProcessor(class_names)
    raw_processor.set_data_path(raw_data_path)
    raw_dataset = raw_processor.load_data()

    filtered_processor = DataProcessor(class_names)
    filtered_processor.set_data_path(filtered_data_path)
    filtered_dataset = filtered_processor.load_data()

    # FeatureExtractor Nesnesini Kullanarak Özellik Çıkar
    extractor = FeatureExtractor(window_size=8)

    print("Filtrelenmemiş veri için özellikler çıkarılıyor...")
    raw_features_df = extractor.extract_features(raw_dataset)

    print("Filtrelenmiş veri için özellikler çıkarılıyor...")
    filtered_features_df = extractor.extract_features(filtered_dataset)

    # Evaluator Nesnesini Kullan
    evaluator = Evaluator()

    # Ham Verilerle Değerlendirme
    print("\nHam veriler üzerinde değerlendirme...")
    raw_X, raw_y = evaluator.preprocess_data(raw_dataset, feature_only=False)
    raw_accuracy, raw_f1, raw_report, raw_cm = evaluator.evaluate(raw_X, raw_y)
    print("Doğruluk (Accuracy):", raw_accuracy)
    print("F1 Skoru:", raw_f1)
    print(raw_report)
    evaluator.plot_confusion_matrix(raw_cm, class_names, title="Confusion Matrix (Raw Data)")

    # Filtrelenmiş Verilerle Değerlendirme
    print("\nFiltrelenmiş veriler üzerinde değerlendirme...")
    filtered_X, filtered_y = evaluator.preprocess_data(filtered_dataset, feature_only=False)
    filtered_accuracy, filtered_f1, filtered_report, filtered_cm = evaluator.evaluate(filtered_X, filtered_y)
    print("Doğruluk (Accuracy):", filtered_accuracy)
    print("F1 Skoru:", filtered_f1)
    print(filtered_report)
    evaluator.plot_confusion_matrix(filtered_cm, class_names, title="Confusion Matrix (Filtered Data)")

    # Filtrelenmemiş Özelliklerle Değerlendirme
    print("\nFiltrelenmemiş özellikler üzerinde değerlendirme...")
    raw_feat_X, raw_feat_y = evaluator.preprocess_data(raw_features_df, feature_only=True)
    raw_feat_accuracy, raw_feat_f1, raw_feat_report, raw_feat_cm = evaluator.evaluate(raw_feat_X, raw_feat_y)
    print("Doğruluk (Accuracy):", raw_feat_accuracy)
    print("F1 Skoru:", raw_feat_f1)
    print(raw_feat_report)
    evaluator.plot_confusion_matrix(raw_feat_cm, class_names, title="Confusion Matrix (Raw Features)")

    # Filtrelenmiş Özelliklerle Değerlendirme
    print("\nFiltrelenmiş özellikler üzerinde değerlendirme...")
    filtered_feat_X, filtered_feat_y = evaluator.preprocess_data(filtered_features_df, feature_only=True)
    filtered_feat_accuracy, filtered_feat_f1, filtered_feat_report, filtered_feat_cm = evaluator.evaluate(filtered_feat_X, filtered_feat_y)
    print("Doğruluk (Accuracy):", filtered_feat_accuracy)
    print("F1 Skoru:", filtered_feat_f1)
    print(filtered_feat_report)
    evaluator.plot_confusion_matrix(filtered_feat_cm, class_names, title="Confusion Matrix (Filtered Features)")

    print("\n--- Tüm değerlendirmeler tamamlandı ---")
