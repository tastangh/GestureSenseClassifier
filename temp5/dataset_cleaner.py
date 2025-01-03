# dataset_cleaner.py
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