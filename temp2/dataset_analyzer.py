import pandas as pd
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Veri setini yükler."""
        self.data = pd.read_csv(self.file_path)
        print(f"Veri seti başarıyla yüklendi: {self.file_path}")

    def basic_info(self):
        """Veri seti hakkında temel bilgileri görüntüler."""
        if self.data is not None:
            print("\nVeri Seti Bilgileri:")
            print(self.data.info())
            print("\nVeri Setinin İlk 5 Satırı:")
            print(self.data.head())
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")

    def check_missing_values(self):
        """Veri setindeki eksik değerleri kontrol eder."""
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print("\nHer Sütundaki Eksik Değer Sayısı:")
            print(missing_values)
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")

    def visualize_class_distribution(self):
        """Sınıf dağılımını grafik ve sayılarla görselleştirir."""
        if self.data is not None:
            if "class" in self.data.columns:
                class_counts = self.data["class"].value_counts().sort_index()
                
                # Bar grafiği çiz
                ax = class_counts.plot(kind="bar", figsize=(10, 6), color="skyblue", edgecolor="black")
                plt.title("Sınıf Dağılımı", fontsize=16)
                plt.xlabel("Sınıf", fontsize=12)
                plt.ylabel("Frekans", fontsize=12)
                
                # Çubukların üzerine sayı etiketleri ekle
                for i, count in enumerate(class_counts):
                    plt.text(i, count + 500, str(count), ha='center', fontsize=10)
                
                plt.xticks(rotation=0)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()
                
                # Dağılımı tablo olarak yazdır
                print("\nSınıf Dağılımı (Sayısal):")
                print(class_counts)
            else:
                print("Veri setinde 'class' sütunu bulunamadı.")
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")


if __name__ == "__main__":
    # Veri seti dosyasının yolu
    file_path = "./dataset/EMG-data.csv"  # Dosya yolunu ayarlayın
    analyzer = DatasetAnalyzer(file_path)

    # Veri analizi gerçekleştir
    analyzer.load_data()
    analyzer.basic_info()
    analyzer.check_missing_values()
    analyzer.visualize_class_distribution()
