import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot

class DataProcessor:
    def __init__(self, data_path, class_names, save_path="results/dataset"):
        """
        :param data_path: Karıştırılmış veri dosyasının yolu
        :param class_names: Sınıf isimlerinin listesi
        :param save_path: Grafiklerin kaydedileceği dizin
        """
        self.data_path = data_path
        self.class_names = class_names
        self.save_path = save_path
        self.dataset = None

        # Klasör oluştur
        os.makedirs(self.save_path, exist_ok=True)

    def load_data(self):
        """Veriyi yükler."""
        if os.path.exists(self.data_path):
            print(f"{self.data_path} yükleniyor...")
            self.dataset = pd.read_csv(self.data_path)
            print("Veri başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"{self.data_path} dosyası bulunamadı!")
        return self.dataset

    def plot_line(self, data_row, class_name):
        """Zaman serisi çizgi grafiği."""
        x_time = np.linspace(5, 40, 8)  # 200 Hz örnekleme
        plt.figure(figsize=(12, 6))
        for i in range(8):  # 8 sensör
            sensor_data = data_row[i::8].values
            plt.plot(x_time, sensor_data, label=f"Sensor {i + 1}")

        plt.title(f"{class_name} Hareketi - Zaman Serisi (Line Plot)")
        plt.xlabel("Zaman (ms)")
        plt.ylabel("Örnek Değeri")
        plt.legend(loc="upper right")
        plt.grid(True)

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_line.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için çizgi grafiği {save_file} konumuna kaydedildi.")

    def plot_histograms(self, data_row, class_name):
        """Her sensör için ayrı histogram."""
        plt.figure(figsize=(16, 8))
        for i in range(8):  # 8 sensör
            sensor_data = data_row[i::8].values
            plt.subplot(2, 4, i + 1)
            plt.hist(sensor_data, bins=10, color='blue', edgecolor='black')
            plt.title(f"Sensor {i + 1}")
            plt.xlabel("Örnek Değeri")
            plt.ylabel("Frekans")
            plt.tight_layout()

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_histograms.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için sensör bazında histogramlar {save_file} konumuna kaydedildi.")

    def plot_densities(self, data_row, class_name):
        """Her sensör için ayrı yoğunluk grafiği."""
        plt.figure(figsize=(12, 6))
        for i in range(8):  # 8 sensör
            sensor_data = pd.Series(data_row[i::8].values)
            sensor_data.plot(kind='kde', label=f"Sensor {i + 1}")

        plt.title(f"{class_name} Hareketi - Yoğunluk Grafikleri")
        plt.xlabel("Örnek Değeri")
        plt.legend(loc="upper right")
        plt.grid(True)

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_densities.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için yoğunluk grafikleri {save_file} konumuna kaydedildi.")

    def plot_box(self, data_row, class_name):
        """Box-and-Whisker Plot."""
        data = data_row.values.reshape(8, 8).T
        plt.figure(figsize=(12, 6))
        plt.boxplot(data, labels=[f"Sensor {i+1}" for i in range(8)], vert=True)
        plt.title(f"{class_name} Hareketi - Box-and-Whisker Plot")
        plt.xlabel("Sensör")
        plt.ylabel("Örnek Değeri")

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_box.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için box plot {save_file} konumuna kaydedildi.")

    def plot_heatmap(self, data_row, class_name):
        """Isı Haritası."""
        heatmap_data = data_row.values.reshape(8, 8)
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap_data, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Örnek Değeri")
        plt.title(f"{class_name} Hareketi - Isı Haritası")
        plt.xlabel("Zaman Noktası")
        plt.ylabel("Sensör Numarası")

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_heatmap.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için ısı haritası {save_file} konumuna kaydedildi.")

    def plot_autocorrelation(self, data_row, class_name):
        """Her sensör için otomatik korelasyon grafiği."""
        plt.figure(figsize=(16, 8))
        for i in range(8):  # 8 sensör
            plt.subplot(2, 4, i + 1)
            sensor_data = pd.Series(data_row[i::8].values)
            autocorrelation_plot(sensor_data, ax=plt.gca())
            plt.title(f"Sensor {i + 1}")
            plt.tight_layout()

        save_file = os.path.join(self.save_path, f"{class_name.lower().replace(' ', '_')}_autocorrelation.png")
        plt.savefig(save_file)
        # plt.show()

        print(f"{class_name} sınıfı için sensör bazında otomatik korelasyon grafikleri {save_file} konumuna kaydedildi.")

    def visualize_each_class(self):
        """Her Gesture_Class için bir örnek seçer ve görselleştirir."""
        if self.dataset is None:
            raise ValueError("Veri seti yüklenmemiş. Önce `load_data` çağrılmalı.")
        
        gesture_classes = self.dataset["Gesture_Class"].unique()
        for gesture_class in gesture_classes:
            sample_row = self.dataset[self.dataset["Gesture_Class"] == gesture_class].iloc[0, :-1]
            class_name = self.class_names[int(gesture_class)]
            print(f"{class_name} sınıfı görselleştiriliyor...")
            self.plot_line(sample_row, class_name)
            # self.plot_histograms(sample_row, class_name)
            # self.plot_densities(sample_row, class_name)
            # self.plot_box(sample_row, class_name)
            self.plot_heatmap(sample_row, class_name)
            self.plot_autocorrelation(sample_row, class_name)

if __name__ == "__main__":
    # Veri dosyasının yolu ve sınıf isimleri
    data_path = "dataset/emg_data.csv"
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # DataProcessor nesnesi oluştur
    processor = DataProcessor(data_path, class_names)

    # Veriyi yükle
    dataset = processor.load_data()

    # Her Gesture_Class için bir örneği görselleştir
    processor.visualize_each_class()

    print("\n--- Tüm görselleştirme işlemleri tamamlandı ---")
