import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class ANNTrainer:
    """
    ANN modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, input_dim, random_state=42):
        """
        ANNTrainer sınıfını başlatır.
        :param input_dim: Giriş boyutu (özellik sayısı)
        :param hidden_layers: Her bir gizli katmandaki nöron sayısını içeren liste
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        """
        self.model = None
        self.input_dim = input_dim
        self.history = None
        self.random_state = random_state

    def build_model(self, hidden_layers, dropout_rate, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layers[0], input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return model

    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, n_trials=10):
         # Random search için hiperparametre aralığı
        param_dist = {
            "hidden_layers": [ [32], [64], [32,32], [64,32],[64,64]  ],
            "dropout_rate": np.arange(0.1, 0.5, 0.1),
            "learning_rate": np.logspace(-4, -2, 5)
        }
        
        # RandomizedSearchCV ile optimizasyon
        ann_random = RandomizedSearchCV(
            estimator=self,
            param_distributions=param_dist,
            n_iter=n_trials,  # Deneme sayısı
            cv=3,
            random_state=self.random_state,
            scoring="accuracy"
        )
        
        ann_random.fit(X_train, y_train)

        print("ANN - En iyi Parametreler:", ann_random.best_params_)
        best_params = ann_random.best_params_
        self.model = self.build_model(hidden_layers=best_params["hidden_layers"], dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"] )


    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, optimize=True, n_trials=10):
        """
        Modeli eğitir ve eğitim/doğrulama metriklerini hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        :param epochs: Eğitim için epoch sayısı
        :param batch_size: Eğitim için batch boyutu
        """
        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        else:
            self.model = self.build_model(hidden_layers=[64,32], dropout_rate=0.2, learning_rate=0.001)

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        print("\nEğitim Tamamlandı!")

    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        predictions = self.model.predict(X_test)
        return (predictions > 0.5).astype(int).flatten()
    
    def fit(self, X, y, **fit_params):
        
        hidden_layers = fit_params['hidden_layers']
        dropout_rate= fit_params['dropout_rate']
        learning_rate = fit_params['learning_rate']
        
        model = self.build_model(hidden_layers, dropout_rate, learning_rate)
        model.fit(X, y, verbose=0)
        self.model = model
        return self