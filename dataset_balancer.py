from imblearn.over_sampling import SMOTE
import pandas as pd

class DatasetBalancer:
    """
    Sınıflar arası dengeyi sağlamak için SMOTE kullanan sınıf.
    """
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)

    def balance(self, data, class_column="class"):
        """
        Veriyi SMOTE ile dengeler.
        :param data: Pandas DataFrame (Ham veri)
        :param class_column: Sınıf etiketi sütununun adı
        :return: Dengelenmiş veri çerçevesi (DataFrame)
        """
        labels = data[class_column]
        features = data.drop(columns=[class_column])
        balanced_features, balanced_labels = self.smote.fit_resample(features, labels)

        # Dengelenmiş veriyi yeniden birleştir
        balanced_data = pd.DataFrame(balanced_features, columns=features.columns)
        balanced_data[class_column] = balanced_labels
        return balanced_data
