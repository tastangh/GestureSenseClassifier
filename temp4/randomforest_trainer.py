from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class RandomForestTrainer:
    """
    Random Forest modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, random_state=42, class_weight="balanced"):
        """
        RandomForestTrainer sınıfını başlatır.
        :param n_estimators: Ağaç sayısı
        :param max_depth: Maksimum derinlik
        :param random_state: Rastgelelik kontrolü
        :param class_weight: Sınıf ağırlıkları ('balanced' veya None)
        """
        self.model = None  # Başlangıçta modeli None olarak ayarlıyoruz
        self.random_state = random_state
        self.class_weight=class_weight
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None
    
    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, n_trials=10):
        # Random search için hiperparametre aralığı
        param_dist = {
            "n_estimators": np.arange(50, 250, 10),
            "max_depth": np.arange(5, 30, 2),
        }
        
        # RandomizedSearchCV ile optimizasyon
        rf_random = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, class_weight=self.class_weight),
            param_distributions=param_dist,
            n_iter=n_trials,  # Deneme sayısı
            cv=3, # Cross validation sayısı
            random_state=self.random_state,
            scoring="accuracy",
        )
        
        rf_random.fit(X_train, y_train)
        
        print("Random Forest - En iyi Parametreler:", rf_random.best_params_)
        self.model = rf_random.best_estimator_
        
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=True, n_trials=10):
        
        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        else:
             self.model = RandomForestClassifier(n_estimators=100,max_depth=None, random_state=self.random_state, class_weight=self.class_weight)
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
        return self.model.predict(X_test)