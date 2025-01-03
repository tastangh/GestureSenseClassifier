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
    def drop_rows_by_class(data, class_column, class_value):
        """
        Belirtilen sınıfa (class) sahip satırları veri setinden kaldırır.
        :param data: pandas DataFrame
        :param class_column: Sınıf sütununun adı
        :param class_value: Kaldırılacak sınıf değeri
        :return: pandas DataFrame
        """
        print(f"{class_column} == {class_value} olan satırlar kaldırılıyor...")
        return data[data[class_column] != class_value].reset_index(drop=True)

    @staticmethod
    def drop_na(data):
        """
        Eksik verileri (NaN) temizler.
        :param data: pandas DataFrame
        :return: pandas DataFrame
        """
        print("Eksik veriler (NaN) temizleniyor...")
        return data.dropna().reset_index(drop=True)
