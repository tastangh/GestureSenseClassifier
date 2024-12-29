import os
import pandas as pd
import numpy as np
from dataset_filter import DatasetFilter
from dataset_cleaner import DatasetCleaner
from dataset_balancer import DatasetBalancer
from dataset_feature_extractor import DatasetFeatureExtractor
from dataset_scaler import DatasetScaler
from log_reg_trainer import LogRegTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main(file_path):
    # Çıktı klasörünü oluştur
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu.")

    # Veri yükleme
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)
    channels = [f"channel{i}" for i in range(1, 9)]

    # Veri temizleme
    print("Veri temizleme işlemi yapılıyor...")
    cleaner = DatasetCleaner()
    data = cleaner.drop_columns(data, columns=["label"])

    # Filtreleme işlemi
    print("Tüm kanallar için band geçiren filtre uygulanıyor...")
    filter_processor = DatasetFilter(data, channels, sampling_rate=1000)
    filter_processor.filter_all_channels(filter_type="band", cutoff=(0.1, 499), order=4)
    filtered_data = filter_processor.get_filtered_data()

    # Grafikler
    print("\nTüm kanallar için frekans spektrumları çiziliyor...")
    frequency_plot_path = os.path.join(output_dir, "frequency_spectra.png")
    filter_processor.plot_frequency_spectrum(
        signals=[data[channel] for channel in channels],
        filtered_signals=[filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Frekans Spektrumu" for channel in channels],
        output_path=frequency_plot_path
    )

    print("\nTüm kanallar için zaman domeni sinyalleri çiziliyor...")
    time_plot_path = os.path.join(output_dir, "time_domain_signals.png")
    filter_processor.plot_signals(
        signals=[data[channel] for channel in channels],
        filtered_signals=[filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Zaman Domeni Sinyalleri" for channel in channels],
        output_path=time_plot_path,
        start=0,
        end=1000
    )

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
    trainer = LogRegTrainer(max_iter=100)
    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # Tahmin ve değerlendirme
    print("Model değerlendiriliyor...")
    y_pred = trainer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

    # Sonuçların kaydedilmesi
    print(f"Model eğitimi ve değerlendirme başarıyla tamamlandı. Sonuçlar '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    dataset_path = "dataset/EMG-data.csv"  # Dataset yolunu güncelleyin
    main(dataset_path)
