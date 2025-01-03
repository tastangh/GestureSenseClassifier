import pandas as pd
from dataset_processor import DatasetProcessor
from emg_signal_processor import EMGSignalProcessor
from emg_feature_extractor import EMGFeatureExtractor
from log_reg_trainer import LogisticRegressionTrainer

def main():
    # Veri yükleme
    file_path = "./dataset/EMG-data.csv"
    print("Veri seti yükleniyor...")
    processor = DatasetProcessor(file_path)
    processor.load_data()

    # 1. Veri işleme
    print("\nAdım 1: Veri İşleme...")
    processor.remove_unmarked_data()  # Sınıf 0 olanları çıkar
    processor.balance_data_with_smote()  # SMOTE ile sınıfları dengele
    processed_data = processor.get_cleaned_data()
    print("Veri işleme tamamlandı.\n")

    # İşlenmiş veriyi kaydetme
    processed_file_path = "./dataset/processed_emg_data.csv"
    processed_data.to_csv(processed_file_path, index=False)
    print(f"İşlenmiş veri kaydedildi: {processed_file_path}\n")

    # 2. Sinyal filtreleme
    print("\nAdım 2: Sinyal Filtreleme...")
    channels = [f"channel{i}" for i in range(1, 9)]
    signal_processor = EMGSignalProcessor(processed_data, channels)
    signal_processor.filter_all_channels(filter_type="band", cutoff=(20, 450), order=4)
    filtered_data = signal_processor.get_filtered_data()
    print("Sinyal filtreleme tamamlandı.\n")

    # Filtrelenmiş veriyi kaydetme
    filtered_file_path = "./dataset/filtered_emg_data.csv"
    filtered_data.to_csv(filtered_file_path, index=False)
    print(f"Filtrelenmiş veri kaydedildi: {filtered_file_path}\n")

    # 3. Özellik çıkarımı
    print("\nAdım 3: Özellik Çıkarımı...")
    feature_extractor = EMGFeatureExtractor(filtered_data, channels, window_size=200, sampling_rate=1000)
    features = feature_extractor.extract_features()
    print("Özellik çıkarımı tamamlandı.\n")

    # Özellik verisini kaydetme
    features_file_path = "./dataset/emg_features.csv"
    features.to_csv(features_file_path, index=False)
    print(f"Özellikler kaydedildi: {features_file_path}\n")

    # 4. Model eğitimi ve değerlendirme
    print("\nAdım 4: Logistic Regression Model Eğitimi ve Değerlendirme...")
    target_column = "class"
    trainer = LogisticRegressionTrainer(features, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.preprocess_data()
    model, val_accuracy, test_accuracy, y_val_pred, y_test_pred = trainer.train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Sonuçları görselleştirme
    trainer.plot_confusion_matrix(y_test, y_test_pred, labels=sorted(features[target_column].unique()),
                                   title="Test Set Confusion Matrix")

    # Sonuçları yazdır
    print("\nSonuçlar:")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()