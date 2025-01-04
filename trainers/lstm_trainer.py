import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import sys

# Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_filter import DatasetFilter
from utils.dataset_cleaner import DatasetCleaner
from utils.dataset_balancer import DatasetBalancer
from utils.dataset_feature_extractor import DatasetFeatureExtractor
from utils.dataset_scaler import DatasetScaler
from utils.save_results import save_results_to_excel

# TensorFlow GPU memory limit configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use 9GB of GPU RAM
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class LSTMTrainer:
    """
    LSTM modeli eğitmek ve metrik grafikleri çizmek için bir sınıf.
    """

    def __init__(self, input_shape, num_classes, lstm_units=50, dropout_rate=0.2, learning_rate=0.001,
                 output_dir="./models"):
        """
        LSTMTrainer sınıfını başlatır.
        :param input_shape: Giriş verisinin şekli (örneğin, (zaman_adımı, özellik_sayısı))
        :param lstm_units: LSTM hücre sayısı
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        :param output_dir: Modelin kaydedileceği klasör
        """
        self.num_classes = num_classes
        self.model = Sequential([
            LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')  # Çoklu sınıflandırma için softmax
        ])
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = CategoricalCrossentropy()
        self.history = None
        self.output_dir = output_dir
        self.label_binarizer = LabelBinarizer()
        os.makedirs(self.output_dir, exist_ok=True)

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
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience // 2, min_lr=min_lr))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = []
            epoch_acc = []
            metric = tf.metrics.Accuracy()  # Define metric
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, y_pred = self.train_step(x_batch_train, y_batch_train)
                metric.update_state(tf.argmax(y_batch_train, axis=1), tf.argmax(y_pred, axis=1))
                acc = metric.result().numpy()
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                print(f"  Batch {step + 1}/{len(train_dataset)}, Loss: {loss:.4f}, Accuracy: {acc:.4f}", end='\r')
            print()  # Her batch'den sonra satır atla
            metric.reset_state()  # reset the metric after each epoch

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
        val_acc_list = []
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

    def save_model(self, filename="lstm_model.h5"):
        """
        Eğitilmiş modeli kaydeder.
        :param filename: Modelin kaydedileceği dosya adı
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, filename)
        self.model.save(file_path)
        print(f"LSTM Modeli kaydedildi: {file_path}")

    def load_model(self, filename="lstm_model.h5"):
        """
        Kaydedilmiş modeli yükler.
        :param filename: Modelin yükleneceği dosya adı
        :return: Yüklenen model
        """
        file_path = os.path.join(self.output_dir, filename)
        self.model = tf.keras.models.load_model(file_path)
        print(f"LSTM Modeli yüklendi: {file_path}")
        return self.model


if __name__ == '__main__':
    # Sample usage
    # TensorFlow GPU memory limit configuration
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use 9GB of GPU RAM
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    dataset_path = "dataset/EMG-data.csv"
    output_dir = "./models"

    print("Veri yükleniyor...")
    data = pd.read_csv(dataset_path)
    channels = [f"channel{i}" for i in range(1, 9)]

    # 1. Veri temizleme işlemi
    print("Veri temizleme işlemi yapılıyor...")
    cleaner = DatasetCleaner()
    data = cleaner.drop_columns(data, columns=["label"])  # Gereksiz kolonları temizle
    data = cleaner.drop_unmarked_class(data, class_column="class", unmarked_value=0)  # Class 0'ı temizle

    # 2. SMOTE ile dengeleme
    print("Veri SMOTE ile dengeleniyor...")
    balancer = DatasetBalancer()
    balanced_data = balancer.balance(data, class_column="class")

    # 3. Filtreleme işlemi
    print("Tüm kanallar için band geçiren filtre uygulanıyor...")
    filter_processor = DatasetFilter(balanced_data, channels, sampling_rate=1000)
    filter_processor.filter_all_channels(filter_type="band", cutoff=(0.1, 499), order=4)
    filtered_data = filter_processor.get_filtered_data()

    # 4. Özellik çıkarımı
    print("Özellikler çıkarılıyor...")
    features, labels = DatasetFeatureExtractor.extract_features(filtered_data, channels)
   
     # 5. Veri ölçeklendirme ve bölme
    print("Veri ölçekleniyor ve bölünüyor...")
    scaler = DatasetScaler()
    X_train_full, X_test, y_train_full, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    time_steps = 8
    total_features = X_train.shape[1] // time_steps

     # X_train ve X_test yeniden şekillendiriliyor
    X_train_lstm = X_train.reshape(-1, time_steps, total_features)
    X_val_lstm = X_val.reshape(-1, time_steps, total_features)
    X_test_lstm = X_test.reshape(-1, time_steps, total_features)
   
    num_classes = len(np.unique(y_train))
    model_params = {"time_steps": 8, "lstm_units": 64, "epochs": 10, "batch_size": 90}
    lstm_trainer = LSTMTrainer(input_shape=(time_steps, total_features), num_classes=num_classes,
                              lstm_units=model_params.get("lstm_units", 64), output_dir=output_dir)
    lstm_trainer.train(
        X_train_lstm, y_train, X_val=X_val_lstm, y_val=y_val,
        epochs=model_params.get("epochs", 10),
        batch_size=model_params.get("batch_size", 32),
        learning_rate=model_params.get("learning_rate", 0.001),
        early_stopping=model_params.get("early_stopping", False),
        patience=model_params.get("patience", 3),
        learning_rate_scheduling=model_params.get("learning_rate_scheduling", False),
        factor=model_params.get("factor", 0.1),
        min_lr=model_params.get("min_lr", 1e-6)
    )

    print("LSTM modeli değerlendiriliyor...")
    y_pred_lstm = lstm_trainer.predict(X_test_lstm)
    accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
    print(f"LSTM Test Doğruluğu: {accuracy_lstm * 100:.2f}%")

    save_results_to_excel(output_dir, "LSTM", model_params,
                        lstm_trainer.history['loss'][-1],
                        lstm_trainer.history['accuracy'][-1],
                        lstm_trainer.history['val_loss'][-1] if 'val_loss' in lstm_trainer.history else None,
                        lstm_trainer.history['val_accuracy'][-1] if 'val_accuracy' in lstm_trainer.history else None,
                        )
    lstm_trainer.plot_metrics("LSTM", model_params)

    lstm_trainer.save_model(filename="lstm_model.h5")  # save model
    