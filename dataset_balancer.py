from imblearn.over_sampling import SMOTE

class DatasetBalancer:
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)

    def balance(self, features, labels):
        return self.smote.fit_resample(features, labels)
