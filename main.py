import pandas as pd
import numpy as np
from dataset_filter import DatasetFilter
from dataset_cleaner import DatasetCleaner
from dataset_balancer import DatasetBalancer
from dataset_feature_extractor import DatasetFeatureExtractor
from dataset_scaler import DatasetScaler
from log_reg_model_trainer import ModelTrainer
from confusion_matrix_plot import ConfusionMatrixPlotter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main(file_path):
    # Veri yükleme
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)
    channels = [f"channel{i}" for i in range(1, 9)]

    # Veri temizleme
    print("Veri temizleme işlemi yapılıyor...")
    cleaner = DatasetCleaner()
    data = cleaner.drop_columns(data, columns=["label"])  # Gereksiz sütunları kaldır
    # İleride kullanılabilecek örnek:
    # data = cleaner.drop_rows_by_class(data, class_column="class", class_value=0)  # class == 0 olanları kaldır

    # Filtreleme işlemi
    print("Tüm kanallar için band geçiren filtre uygulanıyor...")
    filter_processor = DatasetFilter(data, channels, sampling_rate=1000)
    filter_processor.filter_all_channels(filter_type="band", cutoff=(20, 450), order=4)
    filtered_data = filter_processor.get_filtered_data()

    # Özellik çıkarma
    print("Özellikler çıkarılıyor...")
    features, labels = DatasetFeatureExtractor.extract_features(filtered_data, channels)

    # Veri dengeleme
    print("Veri SMOTE ile dengeleniyor...")
    balancer = DatasetBalancer()
    features, labels = balancer.balance(features, labels)

    # Veri ölçekleme
    print("Veri ölçekleniyor...")
    scaler = DatasetScaler()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model eğitimi
    print("Lojistik regresyon modeli eğitiliyor...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    y_pred = trainer.predict(X_test)

    # Değerlendirme
    print("Model değerlendiriliyor...")
    accuracy = accuracy_score(y_test, y_pred)
    ConfusionMatrixPlotter.plot(y_test, y_pred, labels=np.unique(labels))
    print(f"Başarı oranı: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    dataset_path = "dataset/EMG-data.csv"
    main(dataset_path)
