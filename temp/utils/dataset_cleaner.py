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
    def drop_unmarked_class(data, class_column, unmarked_value=0):
        """
        Belirtilen "unmarked" sınıfa sahip satırları veri setinden kaldırır.
        :param data: pandas DataFrame
        :param class_column: Sınıf sütununun adı
        :param unmarked_value: Kaldırılacak "unmarked" sınıf değeri
        :return: pandas DataFrame
        """
        print(f"{class_column} sütununda {unmarked_value} değerine sahip satırlar kaldırılıyor...")
        return data[data[class_column] != unmarked_value].reset_index(drop=True)

    @staticmethod
    def drop_na(data):
        """
        Eksik verileri (NaN) temizler.
        :param data: pandas DataFrame
        :return: pandas DataFrame
        """
        print("Eksik veriler (NaN) temizleniyor...")
        return data.dropna().reset_index(drop=True)
