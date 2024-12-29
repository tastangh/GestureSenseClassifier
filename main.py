import pandas as pd
from dataset_analyzer import DatasetAnalyzer
from dataset_balancer import DatasetBalancer
from dataset_emg_signal_filter import EMGSignalProcessor
# from feature_extractor import FeatureExtractor
from log_reg_trainer import LogisticRegressionTrainer


if __name__ == "__main__":
    # Adım 1: Veri yükleme ve analiz
    # print("Veri analizine başlanıyor...")
    file_path = "./dataset/EMG-data.csv"
    # analyzer = DatasetAnalyzer(file_path)
    # analyzer.load_data()
    # analyzer.basic_info()
    # analyzer.check_missing_values()
    # analyzer.visualize_class_distribution()

    data = pd.read_csv(file_path)
    target_column = "class"

    # Adım 2: Filtreleme
    print("\nFiltreleme işlemi başlıyor...")
    channels = [f"channel{i}" for i in range(1, 9)]  # channel1, channel2, ..., channel8
    processor = EMGSignalProcessor(data, channels)
    processor.filter_all_channels(filter_type="low", cutoff=5,order=4, sampling_rate=1000)

    filtered_data = processor.filtered_data
    # filtered_data_path = "./dataset/filtered_emg_data.csv"
    # filtered_data.to_csv(filtered_data_path, index=False)
    # print(f"Filtrelenmiş veri kaydedildi: {filtered_data_path}")

    # Adım 3: Özellik çıkarma
    # print("\nÖzellik çıkarma işlemi başlıyor...")
    # extractor = FeatureExtractor(filtered_data, channels, target_column)
    # featured_data = extractor.extract_features()
    # featured_data_path = "./dataset/featured_filtered_balanced_emg_data.csv"
    # featured_data.to_csv(featured_data_path, index=False)
    # print(f"Özellik çıkarılmış veri kaydedildi: {featured_data_path}")

    
    # Adım 2: Veri dengeleme
    print("\nVeri dengeleniyor...")
    balancer = DatasetBalancer(filtered_data, target_column)
    balancer.class_distribution()
    balanced_data = balancer.balance()
    # balanced_data_path = "./dataset/featured_filtered_balanced_emg_data.csv"
    # balanced_data.to_csv(balanced_data_path, index=False)
    # print(f"Dengelenmiş veri kaydedildi: {balanced_data_path}")


    # Adım 5: Logistic Regression Model Eğitimi ve Değerlendirme
    print("\nModel eğitimi başlıyor...")
    trainer = LogisticRegressionTrainer(balanced_data, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.preprocess_data()
    results = trainer.train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    # Confusion Matrix Görselleştirme
    trainer.plot_confusion_matrix(results["confusion_matrix"], classes=sorted(balanced_data[target_column].unique()))
