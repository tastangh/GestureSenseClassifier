import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

class LSTMTrainer:
    """
    LSTM modeli eğitmek ve metrik grafikleri çizmek için bir sınıf.
    """
    def __init__(self, input_shape, random_state=42):
        """
        LSTMTrainer sınıfını başlatır.
        :param input_shape: Giriş verisinin şekli (örneğin, (zaman_adımı, özellik_sayısı))
        :param lstm_units: LSTM hücre sayısı
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        """
        self.model = None
        self.input_shape = input_shape
        self.history = None
        self.random_state = random_state

    def build_model(self, lstm_units, dropout_rate, learning_rate):
        model = Sequential([
            LSTM(lstm_units, input_shape=self.input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, n_trials=10):
        param_dist = {
            "lstm_units": np.arange(32, 128, 16),
            "dropout_rate": np.arange(0.1, 0.5, 0.1),
            "learning_rate": np.logspace(-4, -2, 5)
        }

        lstm_random = RandomizedSearchCV(
            estimator=self,
            param_distributions=param_dist,
            n_iter=n_trials,
            cv=3,
            random_state=self.random_state,
            scoring="accuracy",
        )

        lstm_random.fit(X_train, y_train)
        print("LSTM - En iyi Parametreler:", lstm_random.best_params_)
        best_params = lstm_random.best_params_
        self.model = self.build_model(lstm_units=best_params["lstm_units"], dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"])


    
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
            self.model = self.build_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001)
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

    def plot_metrics(self):
        """
        Eğitim ve doğrulama kayıp/doğruluk metriklerini çizer.
        """
        if self.history is None:
            print("Henüz eğitim metrikleri mevcut değil!")
            return

        history = self.history.history

        # Kayıp grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['loss'], label='Eğitim Kaybı')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Doğrulama Kaybı')
        plt.title("Eğitim ve Doğrulama Kaybı")
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp")
        plt.legend()
        plt.show()

        # Doğruluk grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['accuracy'], label='Eğitim Doğruluk')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Doğrulama Doğruluk')
        plt.title("Eğitim ve Doğrulama Doğruluk")
        plt.xlabel("Epoch")
        plt.ylabel("Doğruluk")
        plt.legend()
        plt.show()
    
    def fit(self, X, y, **fit_params):
        
        lstm_units = fit_params['lstm_units']
        dropout_rate= fit_params['dropout_rate']
        learning_rate = fit_params['learning_rate']
        
        model = self.build_model(lstm_units, dropout_rate, learning_rate)
        model.fit(X, y, verbose=0)
        self.model = model
        return self