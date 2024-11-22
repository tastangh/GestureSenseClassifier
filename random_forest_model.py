from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierModel:
    def __init__(self, n_estimators=100):
        """
        RandomForestClassifierModel sınıfını başlatır.
        
        :param n_estimators: Ormandaki ağaç sayısı (varsayılan: 100).
        """
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)

    def train_and_predict(self, X_train, y_train, X_test):
        """
        Modeli eğitir ve test verileri üzerinde tahmin yapar.
        
        :param X_train: Eğitim için kullanılacak özellikler.
        :param y_train: Eğitim için kullanılacak hedef değişken.
        :param X_test: Test için kullanılacak özellikler.
        :return: Tahmin edilen etiketler.
        """
        print("RandomForest eğitim oturumu başladı...\n")
        self.model.fit(X_train, y_train)
        print("Tahminler oluşturuluyor...\n")
        y_pred = self.model.predict(X_test)
        return y_pred
