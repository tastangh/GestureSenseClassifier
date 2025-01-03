import pandas as pd
import matplotlib.pyplot as plt
import os

class DatasetLoader:
    def __init__(self, file_path):
        """
        DatasetLoader sınıfı, bir veri dosyasını yüklemek ve analiz etmek için kullanılır.
        :param file_path: Veri dosyasının yolu
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Veri dosyasını yükler ve DataFrame'e aktarır."""
        try:
            print(f"Veri yükleniyor: {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            print(f"Veri başarıyla yüklendi. Toplam satır sayısı: {len(self.data)}")
        except Exception as e:
            print(f"Veri yüklenirken bir hata oluştu: {e}")
            self.data = None

    def summarize_data(self):
        """Verinin genel özet istatistiklerini ve sütun bilgilerini verir."""
        if self.data is not None:
            print("\n=== Veri Özet Bilgisi ===")
            print(self.data.info())
            print("\n=== İlk 5 Satır ===")
            print(self.data.head())
            print("\n=== Eksik Değerler ===")
            print(self.data.isnull().sum())
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def remove_unmarked_data(self, class_column="class"):
        """
        İşaretlenmemiş veriyi (sınıf 0) veri setinden çıkarır.
        :param class_column: Sınıf sütununun adı
        """
        if self.data is not None:
            original_size = len(self.data)
            self.data = self.data[self.data[class_column] != 0]
            print(f"Sınıf 0 verileri çıkarıldı. Kalan veri sayısı: {len(self.data)} (Önceki: {original_size})")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def plot_class_distribution(self, class_column="class"):
        """
        Veri setindeki sınıf dağılımını çizer ve parametrelerle damgalı şekilde kaydeder.
        :param class_column: Sınıf sütununun adı
        """
        if self.data is not None:
            if class_column in self.data.columns:
                print(f"\nSınıf dağılımı görselleştiriliyor: {class_column}")
                class_counts = self.data[class_column].value_counts()

                plt.figure(figsize=(10, 6))
                bars = class_counts.plot(kind='bar', color='skyblue', alpha=0.8)
                plt.title("Sınıf Dağılımı")
                plt.xlabel("Sınıf")
                plt.ylabel("Örnek Sayısı")
                plt.xticks(rotation=45)

                for idx, count in enumerate(class_counts):
                    plt.text(idx, count + (0.01 * count), f'{count}', ha='center', va='bottom', fontsize=10)

                # Plot dosya adı için parametreleri damgala
                plots_dir = "plots"
                os.makedirs(plots_dir, exist_ok=True)

                class_counts_str = "_".join([f"{cls}_{cnt}" for cls, cnt in class_counts.items()])
                plot_filename = f"class_distribution_{class_counts_str}.png"

                plot_path = os.path.join(plots_dir, plot_filename)
                plt.savefig(plot_path)
                plt.close()  # Grafiği kapat
                print(f"Grafik kaydedildi: {plot_path}")
            else:
                print(f"{class_column} sütunu veri setinde bulunamadı.")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")


# Eğer bu dosya çalıştırılırsa
if __name__ == "__main__":
    # Veri dosyasının yolu
    dataset_path = "dataset/EMG-data.csv"

    loader = DatasetLoader(dataset_path)

    # Veriyi yükle
    loader.load_data()

    # Veriyi özetle
    loader.summarize_data()

    # Sınıf 0'ı çıkar
    loader.remove_unmarked_data()

    # Sınıf dağılımını görselleştir
    loader.plot_class_distribution(class_column="class")
