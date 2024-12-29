import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


class DatasetProcessor:
    def __init__(self, file_path):
        """
        DatasetProcessor sınıfını başlatır.
        :param file_path: Veri seti dosyasının yolu
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Veriyi yükler.
        """
        self.data = pd.read_csv(self.file_path)
        print(f"Veri seti başarıyla yüklendi: {self.file_path}")

    def remove_unmarked_data(self):
        """
        Unmarked verileri (class 0) kaldırır.
        """
        initial_count = len(self.data)
        self.data = self.data[self.data["class"] != 0]
        removed_count = initial_count - len(self.data)
        print(f"Unmarked (class 0) veriler kaldırıldı. {removed_count} örnek çıkarıldı.")

    def balance_data_with_smote(self):
        """
        SMOTE kullanarak veri setini dengeler.
        """
        X = self.data.drop(columns=["class"])
        y = self.data["class"]

        # SMOTE ile veri dengeleme
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Dengelenmiş veriyi yeniden birleştir
        self.data = pd.DataFrame(X_resampled, columns=X.columns)
        self.data["class"] = y_resampled
        print("SMOTE ile veri dengelendi.")
        print("Dengelenmiş veri seti sınıf dağılımı:")
        print(self.data["class"].value_counts())

    def visualize_class_distribution(self):
        """
        Sınıf dağılımını grafik ve sayılarla görselleştirir.
        """
        class_counts = self.data["class"].value_counts().sort_index()
        ax = class_counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Sınıf Dağılımı", fontsize=16)
        plt.xlabel("Sınıf", fontsize=12)
        plt.ylabel("Frekans", fontsize=12)

        # Her bir çubuğun üstüne frekans değerini ekle
        for i, count in enumerate(class_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def get_cleaned_data(self):
        """
        Temizlenmiş ve dengelenmiş veriyi döndürür.
        :return: Pandas DataFrame
        """
        return self.data


if __name__ == "__main__":
    # Veri seti dosyasının yolu
    file_path = "./dataset/EMG-data.csv"
    processor = DatasetProcessor(file_path)

    # Veri yükleme
    processor.load_data()

    # Unmarked verileri kaldırma
    processor.remove_unmarked_data()

    # Sınıf dağılımını görselleştirme (Unmarked kaldırıldıktan sonra)
    print("\nSınıf dağılımı görselleştiriliyor...")
    processor.visualize_class_distribution()

    # SMOTE ile veri dengeleme
    processor.balance_data_with_smote()

    # Sınıf dağılımını görselleştirme (SMOTE sonrası)
    print("\nDengelenmiş sınıf dağılımı görselleştiriliyor...")
    processor.visualize_class_distribution()

    # Temizlenmiş ve dengelenmiş veriyi kaydetme
    cleaned_file_path = "./dataset/cleaned_balanced_emg_data.csv"
    processor.data.to_csv(cleaned_file_path, index=False)
    print(f"Temizlenmiş ve dengelenmiş veri seti kaydedildi: {cleaned_file_path}")
