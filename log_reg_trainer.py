import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

class LogRegTrainer:
    
    """
    Lojistik Regresyon model eğitimi ve metrik grafiği çizimi için sınıf.
    """
    
    def __init__(self, random_state=42):
        """
        LogRegTrainer sınıfını başlatır.
        :param max_iter: Maksimum iterasyon sayısı
        :param random_state: Rastgelelik kontrolü için sabit bir değer
        :param class_weight: Sınıf ağırlıkları ('balanced' veya None)
        """
        self.model = None
        self.random_state= random_state
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None
    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, cv=3):
         # Grid search için hiperparametre aralığı
        param_grid = {
            "C": np.logspace(-4, 4, 10),  # C için daha odaklanmış bir aralık
            "max_iter": np.arange(100, 1500, 200)  # max_iter için daha odaklanmış aralık
        }
        
        # GridSearchCV ile optimizasyon
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=self.random_state, class_weight="balanced"),
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
        )
        
        lr_grid.fit(X_train, y_train)
        
        print("LogisticRegression - En iyi Parametreler:", lr_grid.best_params_)
        self.model = lr_grid.best_estimator_

    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=True, cv=3):
        """
        Modeli eğitir ve eğitim/doğrulama metriklerini hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        """
        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val, cv=cv)
        else:
             self.model = LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight="balanced")
             self.model.fit(X_train, y_train)

        unique_classes = np.unique(y_train)

        # Eğitim metriklerini hesapla
        y_train_pred_prob = self.model.predict_proba(X_train)
        self.train_loss = log_loss(y_train, y_train_pred_prob, labels=unique_classes)
        self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))

        # Doğrulama metriklerini hesapla
        if X_val is not None and y_val is not None:
            y_val_pred_prob = self.model.predict_proba(X_val)
            self.val_loss = log_loss(y_val, y_val_pred_prob, labels=unique_classes)
            self.val_accuracy = accuracy_score(y_val, self.model.predict(X_val))

        # Loglama
        print("\nEğitim Tamamlandı!")
        print(f"Eğitim Kaybı: {self.train_loss:.4f} | Eğitim Doğruluk: {self.train_accuracy:.4f}")
        if self.val_loss is not None:
            print(f"Doğrulama Kaybı: {self.val_loss:.4f} | Doğrulama Doğruluk: {self.val_accuracy:.4f}")

    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        return self.model.predict(X_test)