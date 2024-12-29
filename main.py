import pandas as pd
from dataset_processor import DatasetProcessor
from emg_signal_processor import EMGSignalProcessor
from emg_feature_extractor import EMGFeatureExtractor
from log_reg_trainer import LogisticRegressionTrainer

def main():
    # Veri yükleme
    file_path = "./dataset/EMG-data.csv"
    data = pd.read_csv(file_path)
    print("Veri seti başarıyla yüklendi.")

    # 1. Veri işleme
    print("\nAdım 1: Veri İşleme...")
    processor = DatasetProcessor(data)
    processor.remove_unmarked_data()  # Sınıf 0 olanları çıkar
    processor.balance_classes_with_smote()  # SMOTE ile sınıfları dengele
    processed_data = processor.get_processed_data()
    print("Veri işleme tamamlandı.\n")

    # 2. Sinyal filtreleme
    print("\nAdım 2: Sinyal Filtreleme...")
    channels = [f"channel{i}" for i in range(1, 9)]
    signal_processor = EMGSignalProcessor(processed_data, channels)
    signal_processor.filter_all_channels(filter_type="band", cutoff=(20, 450), order=4)  # Band-pass filtre
    filtered_data = signal_processor.get_filtered_data()
    print("Sinyal filtreleme tamamlandı.\n")

    # 3. Özellik çıkarımı
    print("\nAdım 3: Özellik Çıkarımı...")
    feature_extractor = EMGFeatureExtractor(filtered_data, channels, window_size=200, sampling_rate=1000)
    features = feature_extractor.extract_features()
    print("Özellik çıkarımı tamamlandı.\n")

    # 4. Model eğitimi ve değerlendirme
    print("\nAdım 4: Logistic Regression Model Eğitimi ve Değerlendirme...")
    target_column = "class"
    trainer = LogisticRegressionTrainer(features, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.preprocess_data()
    results = trainer.train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    # Sonuçları yazdır
    print("\nSonuçlar:")
    print(f"Validation Accuracy: {results['val_accuracy']:.2f}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}")

    # Confusion Matrix Görselleştirme
    trainer.plot_confusion_matrix(results['confusion_matrix'], classes=sorted(features[target_column].unique()))

    # Filtrelenmiş ve çıkarılmış özellikli veriyi kaydet
    features_output_path = "./dataset/emg_features_processed.csv"
    features.to_csv(features_output_path, index=False)
    print(f"İşlenmiş özellikler kaydedildi: {features_output_path}")

if __name__ == "__main__":
    main()
