# ann_trainer.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import CategoricalCrossentropy
import os
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt  
class ANNTrainer:
    """
    ANN modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, input_dim, num_classes, hidden_layers, dropout_rate=0.2, learning_rate=0.001, output_dir="./output"):
        """
        ANNTrainer sınıfını başlatır.
        :param input_dim: Giriş boyutu (özellik sayısı)
        :param num_classes: Sınıf sayısı
        :param hidden_layers: Her bir gizli katmandaki nöron sayısını içeren liste
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        :param output_dir: Çıktı dizini
        """
        self.num_classes = num_classes
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.optimizer = None
        self.loss_fn = CategoricalCrossentropy()
        self.history = None
        self.output_dir = output_dir
        self.label_binarizer = LabelBinarizer()

    def _create_optimizer(self, optimizer_type, learning_rate):
            if optimizer_type == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_type == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Invalid optimizer type: {optimizer_type}")


    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32,
               optimizer_type='adam', learning_rate=0.001,
              early_stopping=False, patience=3,
              learning_rate_scheduling=False, factor=0.1, min_lr=1e-6):
        """
        Modeli eğitir ve eğitim/doğrulama metriklerini hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        :param epochs: Eğitim için epoch sayısı
        :param batch_size: Eğitim için batch boyutu
        :param optimizer_type: Kullanılacak optimizasyon algoritması ('adam', 'sgd')
        :param learning_rate: Öğrenme oranı
        :param early_stopping: Early stopping yapılıp yapılmayacağı (bool)
        :param patience: Early stopping sabrı (int)
        :param learning_rate_scheduling: Learning rate scheduling yapılıp yapılmayacağı (bool)
        :param factor: Learning rate'i düşürme faktörü
        :param min_lr: Learning rate için minimum değer
        """
         # One-hot encode labels
        y_train_encoded = self.label_binarizer.fit_transform(y_train)
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_binarizer.transform(y_val)
        else:
            y_val_encoded = None
        
        self.optimizer = self._create_optimizer(optimizer_type, learning_rate)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_encoded)).batch(batch_size)
        if X_val is not None and y_val is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_encoded)).batch(batch_size)
        else:
            val_dataset = None
            
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Callbacks
        callbacks = []
        if early_stopping:
           callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
        if learning_rate_scheduling:
          callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience//2, min_lr=min_lr))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = []
            epoch_acc = []
            metric = tf.metrics.Accuracy() # Define metric
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, y_pred = self.train_step(x_batch_train, y_batch_train)
                metric.update_state(tf.argmax(y_batch_train, axis=1), tf.argmax(y_pred, axis=1))
                acc = metric.result().numpy()
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                print(f"  Batch {step + 1}/{len(train_dataset)}, Loss: {loss:.4f}, Accuracy: {acc:.4f}", end='\r')
            print() # Her batch'den sonra satır atla
            metric.reset_state() # reset the metric after each epoch

            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            print(f"  Eğitim Loss: {avg_loss:.4f}, Eğitim Accuracy: {avg_acc:.4f}")
            if val_dataset is not None:
              val_loss, val_acc = self.validate(val_dataset)
              history['val_loss'].append(val_loss)
              history['val_accuracy'].append(val_acc)
              print(f"  Doğrulama Loss: {val_loss:.4f}, Doğrulama Accuracy: {val_acc:.4f}")


        self.history = history
        print("\nEğitim Tamamlandı!")

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    def validate(self, val_dataset):
        val_loss_list = []
        val_acc_list = [] # This line was missing.
        metric = tf.metrics.Accuracy()
        for x_batch_val, y_batch_val in val_dataset:
          y_pred = self.model(x_batch_val, training=False)
          val_loss = self.loss_fn(y_batch_val, y_pred)
          metric.update_state(tf.argmax(y_batch_val, axis=1), tf.argmax(y_pred, axis=1))
          val_loss_list.append(val_loss)
          val_acc_list.append(metric.result().numpy())
        metric.reset_state()
        return np.mean(val_loss_list), np.mean(val_acc_list)


    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def plot_metrics(self, model_name, model_params):
        """
        Eğitim ve doğrulama kayıp/doğruluk metriklerini çizer ve kaydeder.
        :param model_name: Modelin adı
        :param model_params: Modelin parametreleri (sözlük formatında)
        """
        if self.history is None:
            print("Henüz eğitim metrikleri mevcut değil!")
            return

        history = self.history
        
        # Parametreleri dosya adına dahil et
        params_str = "_".join([f"{key}-{value}" for key, value in model_params.items()])

        # Kayıp grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['loss'], label='Eğitim Kaybı')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Doğrulama Kaybı')
        plt.title("Eğitim ve Doğrulama Kaybı")
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp")
        plt.legend()
        loss_plot_path = os.path.join(self.output_dir, f"{model_name}_loss_{params_str}.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # Doğruluk grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['accuracy'], label='Eğitim Doğruluk')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Doğrulama Doğruluk')
        plt.title("Eğitim ve Doğrulama Doğruluk")
        plt.xlabel("Epoch")
        plt.ylabel("Doğruluk")
        plt.legend()
        accuracy_plot_path = os.path.join(self.output_dir, f"{model_name}_accuracy_{params_str}.png")
        plt.savefig(accuracy_plot_path)
        plt.close()