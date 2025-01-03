import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_balancer import DatasetBalancer
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetProcessor:
    def __init__(self, data_path, test_size=0.2, val_size=0.2, random_state=42, 
                 window_size=200, overlap=0.5,
                 balance_data=True, apply_filter=True, extract_features=True, normalize_data=True,
                 filter_params = {"bandpass_params": {"lowcut": 20, "highcut": 450, "order": 4},
                                  "notch_params": {"freq": 50, "quality": 30}},
                feature_params = {"selected_features": [
                                    "mean_absolute_value", "rms", "waveform_length",
                                    "slope_sign_changes", "willison_amplitude", "dominant_frequency",
                                    "maximum_fractal_length", "log_detector"
                                  ],
                                  "feature_weights": {
                                    "mean_absolute_value": 1.5,
                                    "rms": 1.5,
                                    "waveform_length": 1.3,
                                    "slope_sign_changes": 1.2,
                                    "willison_amplitude": 1.2,
                                    "dominant_frequency": 1.0,
                                    "maximum_fractal_length": 0.8,
                                    "log_detector": 0.7
                                 }}):
        """
        Veri setini yükler, işler ve eğitim-test kümelerine ayırır.
        :param data_path: Veri setinin yolu
        :param test_size: Test kümesinin oranı (0-1 arası)
        :param val_size: Validasyon kümesinin oranı (0-1 arası)
        :param random_state: Rastgelelik için seed değeri
        :param window_size: Özellik çıkarımı için pencere boyutu
        :param overlap: Özellik çıkarımı için örtüşme oranı
        :param balance_data: Sınıf dengelenmesi yapılacak mı
        :param apply_filter: Filtre uygulanacak mı
        :param extract_features: Özellik çıkarımı yapılacak mı
        :param normalize_data: Veri normalize edilecek mi
        :param filter_params: Filtre parametreleri
        :param feature_params: Özellik çıkarma parametreleri
        """
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.window_size = window_size
        self.overlap = overlap
        self.balance_data = balance_data
        self.apply_filter = apply_filter
        self.extract_features = extract_features
        self.normalize_data = normalize_data
        self.filter_params = filter_params
        self.feature_params = feature_params
        self.data = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.scaler = None

        self.channels = [f"channel{i}" for i in range(1, 9)]  # Kanal isimleri


    def load_data(self):
        """Veri setini yükler."""
        print(f"Veri yükleniyor: {self.data_path}")
        loader = DatasetLoader(self.data_path)
        loader.load_data()
        if loader.data is None:
            raise ValueError("Veri yüklenemedi.")

        # Sınıf 0'ı çıkar ve label sütununu sil
        loader.remove_unmarked_data()
        loader.drop_label_column()
        self.data = loader.data
        print("Veri başarıyla yüklendi.")

    def analyze_data(self, class_column="class", output_dir="plots"):
         """Veri setini inceler ve grafikler oluşturur."""
         if self.data is not None:
            os.makedirs(output_dir, exist_ok=True)

            print("\n=== Veri Seti Analizi ===")

            # Eksik değer kontrolü
            print("\nEksik Değerler:")
            print(self.data.isnull().sum())
        
            # Sınıf dağılımı
            print("\nSınıf Dağılımı:")
            class_counts = self.data[class_column].value_counts()
            print(class_counts)
            plt.figure(figsize=(8, 5))
            bars = class_counts.plot(kind="bar", color="skyblue", alpha=0.8)
            plt.title("Sınıf Dağılımı")
            plt.xlabel("Sınıf")
            plt.ylabel("Örnek Sayısı")
            plt.xticks(rotation=45)
            for idx, count in enumerate(class_counts):
                plt.text(idx, count + (0.01 * count), f"{count}", ha="center", va="bottom", fontsize=10)
            plt.tight_layout()
            class_dist_path = os.path.join(output_dir, "class_distribution.png")
            plt.savefig(class_dist_path)
            plt.close()
            print(f"Sınıf dağılımı grafiği kaydedildi: {class_dist_path}")


         else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")
    
    def balance(self, class_column="class"):
        """Veriyi dengeler."""
        if self.balance_data:
            print("Veri dengeleniyor...")
            balancer = DatasetBalancer(target_size=250000, random_state=self.random_state)
            self.data = balancer.balance_data(self.data, class_column=class_column)
            balancer.plot_class_distribution(self.data, class_column=class_column)
            print("Veri dengelendi.")
        else:
            print("Veri dengelenmesi atlandı.")

    def filter(self):
        """Veriyi filtreler."""
        if self.apply_filter:
            print("Veri filtreleniyor...")
            data_filter = DataFilter(self.data, sampling_rate=1000)
            data_filter.apply_filters(columns=self.channels, filter_type="both", **self.filter_params)
            self.data = data_filter.data
            print("Veri filtrelendi.")
        else:
            print("Veri filtreleme atlandı.")
    
    def split_data(self, class_column="class"):
         """Veriyi eğitim, test, validasyon kümelerine ayırır."""
         if self.data is not None:
            # Veriyi eğitim, test ve validasyon kümelerine ayır
            X = self.data[self.channels]
            y = self.data[class_column]
        
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )

            if self.val_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                 X_train, y_train, test_size=self.val_size/(1-self.test_size), stratify=y_train, random_state=self.random_state
                )
                self.train_data = pd.concat([X_train, y_train], axis=1)
                self.test_data = pd.concat([X_test, y_test], axis=1)
                self.val_data = pd.concat([X_val, y_val], axis=1)
                
                print(f"Veri seti başarıyla bölündü: {len(self.train_data)} eğitim, {len(self.val_data)} validasyon, {len(self.test_data)} test örneği.")


            else:
                self.train_data = pd.concat([X_train, y_train], axis=1)
                self.test_data = pd.concat([X_test, y_test], axis=1)
                print(f"Veri seti başarıyla bölündü: {len(self.train_data)} eğitim, {len(self.test_data)} test örneği.")
         else:
              print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def extract_feature(self, class_column="class"):
        """Özellikleri çıkarır."""
        if self.extract_features:
            print("Özellikler çıkarılıyor...")
            
            # Özellik çıkarıcıyı oluştur
            feature_extractor = FeatureExtractor(
                data=None,  # Ham veri artık burada kullanılmıyor
                window_size=self.window_size,
                overlap=self.overlap,
                selected_features=self.feature_params["selected_features"],
                feature_weights=self.feature_params["feature_weights"]
            )

            # Özellikleri her küme için ayrı ayrı çıkar
            self.train_data = pd.concat([feature_extractor.extract_features(data=self.train_data), self.train_data[class_column]], axis=1)
            if self.val_data is not None:
              self.val_data = pd.concat([feature_extractor.extract_features(data=self.val_data), self.val_data[class_column]], axis=1)

            self.test_data = pd.concat([feature_extractor.extract_features(data=self.test_data), self.test_data[class_column]], axis=1)
            print("Özellikler çıkarıldı.")
        else:
            print("Özellik çıkarma atlandı.")
    
    def extract_and_split(self, class_column="class"):
        """Veriyi böler ve özellikleri çıkarır."""
        if self.data is not None:
            # Veriyi eğitim, test ve validasyon kümelerine ayır
            X = self.data[self.channels]
            y = self.data[class_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )

            if self.val_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=self.val_size/(1-self.test_size), stratify=y_train, random_state=self.random_state
                )
                 # Özellik çıkarıcıyı oluştur
                feature_extractor = FeatureExtractor(
                    data=None, # Ham veri artık burada kullanılmıyor
                    window_size=self.window_size,
                    overlap=self.overlap,
                    selected_features=self.feature_params["selected_features"],
                    feature_weights=self.feature_params["feature_weights"]
                )

                # Özellikleri her küme için ayrı ayrı çıkar
                self.train_data = pd.concat([feature_extractor.extract_features(data=pd.concat([X_train, y_train], axis=1)), y_train.reset_index(drop=True)], axis=1)
                self.test_data = pd.concat([feature_extractor.extract_features(data=pd.concat([X_test, y_test], axis=1)), y_test.reset_index(drop=True)], axis=1)
                self.val_data = pd.concat([feature_extractor.extract_features(data=pd.concat([X_val, y_val], axis=1)), y_val.reset_index(drop=True)], axis=1)
                
                print(f"Veri seti başarıyla bölündü ve özellikleri çıkarıldı: {len(self.train_data)} eğitim, {len(self.val_data)} validasyon, {len(self.test_data)} test örneği.")


            else:
                 # Özellik çıkarıcıyı oluştur
                feature_extractor = FeatureExtractor(
                    data=None,
                    window_size=self.window_size,
                    overlap=self.overlap,
                    selected_features=self.feature_params["selected_features"],
                    feature_weights=self.feature_params["feature_weights"]
                )
                # Özellikleri her küme için ayrı ayrı çıkar
                self.train_data = pd.concat([feature_extractor.extract_features(data=pd.concat([X_train, y_train], axis=1)), y_train.reset_index(drop=True)], axis=1)
                self.test_data = pd.concat([feature_extractor.extract_features(data=pd.concat([X_test, y_test], axis=1)), y_test.reset_index(drop=True)], axis=1)
                print(f"Veri seti başarıyla bölündü ve özellikleri çıkarıldı: {len(self.train_data)} eğitim, {len(self.test_data)} test örneği.")
        else:
              print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def normalize(self, class_column="class"):
        """Eğitim verisi üzerinde normalizasyon uygular ve diğer verileri dönüştürür."""
        if self.normalize_data:
            print("Veri normalize ediliyor...")

            if self.train_data is not None:
                # Eğitim verisi üzerinde Min-Max Scaler'ı fit et
                X_train = self.train_data.drop(columns=[class_column])
                self.scaler = MinMaxScaler()
                self.scaler.fit(X_train)

                # Diğer veri setlerini dönüştür
                if self.val_data is not None:
                    X_val = self.val_data.drop(columns=[class_column])
                    self.val_data[X_val.columns] = self.scaler.transform(X_val)
                if self.test_data is not None:
                    X_test = self.test_data.drop(columns=[class_column])
                    self.test_data[X_test.columns] = self.scaler.transform(X_test)

                 # Eğitim verisini dönüştür
                self.train_data[X_train.columns] = self.scaler.transform(X_train)


                print("Veri seti başarıyla normalize edildi.")
                print(f"Normalizasyon parametreleri eğitim verisi üzerinde fit edildi.")
            else:
                print("Normalizasyon için eğitim verisi bulunamadı.")
        else:
              print("Veri normalizasyonu atlandı.")


    def save_data(self, output_dir="dataset"):
        """Bölünmüş ve işlenmiş veri setlerini kaydeder."""
        os.makedirs(output_dir, exist_ok=True)

        if self.train_data is not None:
            train_path = os.path.join(output_dir, "train_data.csv")
            self.train_data.to_csv(train_path, index=False)
            print(f"Eğitim verisi kaydedildi: {train_path}")

        if self.test_data is not None:
            test_path = os.path.join(output_dir, "test_data.csv")
            self.test_data.to_csv(test_path, index=False)
            print(f"Test verisi kaydedildi: {test_path}")

        if self.val_data is not None:
            val_path = os.path.join(output_dir, "val_data.csv")
            self.val_data.to_csv(val_path, index=False)
            print(f"Validasyon verisi kaydedildi: {val_path}")

    def process(self, class_column="class", output_dir="dataset"):
        """Tüm işlem adımlarını yürütür."""
        self.load_data()
        self.analyze_data(class_column=class_column, output_dir=output_dir)
        self.balance(class_column=class_column)
        self.filter()
        self.extract_and_split(class_column=class_column)
        self.normalize(class_column=class_column)
        self.save_data(output_dir=output_dir)



if __name__ == "__main__":
    # Veri setinin yolu (filtrelenmiş ve dengelenmiş veri)
    dataset_path = "dataset/EMG-data.csv"

    # DatasetProcessor nesnesi oluştur
    processor = DatasetProcessor(data_path=dataset_path, 
                                 test_size=0.2, 
                                 val_size=0.2,
                                  balance_data=True,
                                 apply_filter=True,
                                 extract_features=True,
                                 normalize_data=True)

    # Tüm işlem adımlarını yürüt
    processor.process(class_column="class", output_dir="dataset")