import pandas as pd

class DatasetCleaner:
    """
    Veri seti temizleme işlemleri için sınıf.
    """
    @staticmethod
    def drop_columns(data, columns):
        """
        Belirtilen sütunları veri setinden kaldırır.
        :param data: pandas DataFrame
        :param columns: Kaldırılacak sütunların listesi
        :return: pandas DataFrame
        """
        print(f"Gereksiz sütunlar kaldırılıyor: {columns}")
        return data.drop(columns=columns, errors="ignore")
    
    @staticmethod
    def drop_classes(data, classes):
         """
        Belirtilen sınıfları veri setinden kaldırır.
        :param data: pandas DataFrame
        :param classes: Kaldırılacak sınıfların listesi
        :return: pandas DataFrame
         """
         print(f"Belirtilen sınıflar kaldırılıyor: {classes}")
         return data[~data['class'].isin(classes)]