import pandas as pd
from sklearn.utils import resample


class DatasetBalancer:
    def __init__(self, data, target_column):
        """
        DatasetBalancer sınıfını başlatır.
        :param data: Dengelenecek veri seti (pandas DataFrame)
        :param target_column: Sınıf etiketlerini içeren sütun adı
        """
        self.data = data
        self.target_column = target_column

    def class_distribution(self):
        """
        Veri setindeki sınıf dağılımını görüntüler.
        """
        class_counts = self.data[self.target_column].value_counts().sort_index()
        print("\nSınıf Dağılımı (Mevcut):")
        print(class_counts)

    def balance(self, target_size=250000):
        """
        Hem alt örnekleme hem de üst örnekleme kullanarak veri setini dengeler.
        :param target_size: Her sınıf için hedeflenen örnek sayısı
        :return: Dengelenmiş veri seti (pandas DataFrame)
        """
        balanced_data = []
        for class_label in self.data[self.target_column].unique():
            class_data = self.data[self.data[self.target_column] == class_label]
            
            # Alt örnekleme
            if len(class_data) > target_size:
                class_data = class_data.sample(n=target_size, random_state=42)
            
            # Üst örnekleme
            elif len(class_data) < target_size:
                class_data = resample(
                    class_data,
                    replace=True,
                    n_samples=target_size,
                    random_state=42
                )
            
            balanced_data.append(class_data)

        # Dengelenmiş veri setini birleştir
        balanced_data = pd.concat(balanced_data)
        return balanced_data


if __name__ == "__main__":
    # Veri setinin yolu
    file_path = "./dataset/EMG-data.csv"
    
    # Veri setini yükle
    data = pd.read_csv(file_path)
    print(f"Veri seti başarıyla yüklendi: {file_path}")
    
    # DatasetBalancer sınıfını başlat
    balancer = DatasetBalancer(data, target_column="class")
    
    # Mevcut sınıf dağılımını kontrol et
    balancer.class_distribution()
    
    # Veri setini dengeli hale getir
    target_size = 250000  # Her sınıf için hedeflenen örnek sayısı
    balanced_data = balancer.balance(target_size=target_size)
    
    # Dengelenmiş sınıf dağılımını görüntüle
    print("\nDengelenmiş Sınıf Dağılımı:")
    print(balanced_data["class"].value_counts())
    
    # Dengelenmiş veri setini kaydet
    balanced_data.to_csv("./dataset/Balanced_EMG_data.csv", index=False)
    print("\nDengelenmiş veri seti 'Balanced_EMG_data.csv' olarak kaydedildi.")
