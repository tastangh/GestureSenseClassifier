import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import log_loss
import warnings
import time
import multiprocessing

warnings.filterwarnings("ignore")  # Uyarıları kapat

class LstmTrainer:
    """
    LSTM modelini eğitmek ve değerlendirmek için kullanılan sınıf.
    """
    def __init__(self, file_path, channels, window_size=200, cutoff=(20, 450), sampling_rate=1000, base_dir="results", patience=20, lstm_units=50, dropout_rate=0.5, learning_rate=0.001):
        """
        Sınıfın başlatıcı metodu.

        Parametreler:
            file_path (str): Veri setinin dosya yolu.
            channels (list): Kullanılacak kanal isimlerinin listesi.
            window_size (int): Özellik çıkarımı için pencere boyutu.
            cutoff (tuple): Filtreleme için kesim frekansları (alt, üst).
            sampling_rate (int): Örnekleme frekansı.
            base_dir (str): Sonuçların kaydedileceği temel dizin.
            patience (int): Erken durdurma için sabır (epoch sayısı).
            lstm_units (int): LSTM katmanındaki nöron sayısı.
            dropout_rate (float): Dropout oranı.
            learning_rate (float): Öğrenme oranı.
        """
        self.file_path = file_path
        self.channels = channels
        self.window_size = window_size
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.base_dir = base_dir
        self.patience = patience
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate


    def log(self, message, log_file):
        """
        Log mesajlarını hem konsola yazdıran hem de dosyaya kaydeden metot.
        
        Parametreler:
            message (str): Log mesajı.
            log_file (str): Log dosyasının yolu.
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    @staticmethod
    def drop_columns(data, columns):
        """
        Veri setinden belirtilen sütunları silen statik metot.
        
        Parametreler:
            data (pd.DataFrame): Veri seti.
            columns (list): Silinecek sütun isimlerinin listesi.
        
        Returns:
            pd.DataFrame: Sütunları silinmiş veri seti.
        """
        return data.drop(columns=columns, errors="ignore")

    @staticmethod
    def drop_unmarked_class(data, class_column, unmarked_value=0):
        """
        Veri setinden belirtilen sınıfa ait olmayan verileri silen statik metot.

        Parametreler:
            data (pd.DataFrame): Veri seti.
            class_column (str): Sınıf sütununun adı.
            unmarked_value (int): Silinecek sınıf değeri.
        
        Returns:
            pd.DataFrame: İşaretlenmemiş sınıfı silinmiş veri seti.
        """
        return data[data[class_column] != unmarked_value].reset_index(drop=True)

    @staticmethod
    def drop_na(data):
        """
        Veri setinden eksik verileri (NaN) silen statik metot.
        
        Parametreler:
            data (pd.DataFrame): Veri seti.
        
        Returns:
            pd.DataFrame: Eksik verileri silinmiş veri seti.
        """
        return data.dropna().reset_index(drop=True)

    @staticmethod
    def apply_filter(signal, filter_type, cutoff, order=4, sampling_rate=1000):
        """
        Sinyale belirtilen tipte filtre uygulayan statik metot.
        
        Parametreler:
            signal (np.array): Filtrelenecek sinyal.
            filter_type (str): Filtre tipi ("band").
            cutoff (tuple): Kesim frekansları (alt, üst).
            order (int): Filtre derecesi.
            sampling_rate (int): Örnekleme frekansı.
        
        Returns:
           np.array: Filtrelenmiş sinyal.
        """
        nyquist = 0.5 * sampling_rate
        if filter_type == "band":
            normalized_cutoff = [freq / nyquist for freq in cutoff]
            b, a = butter(order, normalized_cutoff, btype="band")
        else:
            raise ValueError("Sadece bant geçiş filtresi destekleniyor.")
        return filtfilt(b, a, signal)

    def filter_all_channels(self, data, use_filter=True):
        """
        Veri setindeki tüm kanallara filtre uygulayan metot.

        Parametreler:
            data (pd.DataFrame): Veri seti.
            use_filter (bool): Filtreleme kullanılacak mı (True/False).

        Returns:
            pd.DataFrame: Filtrelenmiş veri seti.
        """
        if use_filter:
            for channel in self.channels:
                data[channel] = self.apply_filter(data[channel].values, "band", self.cutoff, order=4, sampling_rate=self.sampling_rate)
        return data

    def update_cutoff(self, cutoff):
        """
        Filtre kesim frekanslarını güncelleyen metot.
        
        Parametreler:
             cutoff (tuple): Yeni kesim frekansları.
        """
        self.cutoff = cutoff


    @staticmethod
    def balance_data_with_smote(data, class_column, use_smote=True):
        """
        SMOTE kullanarak veri setini dengeleyen statik metot.

        Parametreler:
            data (pd.DataFrame): Veri seti.
            class_column (str): Sınıf sütununun adı.
            use_smote (bool): SMOTE kullanılacak mı (True/False).

        Returns:
            pd.DataFrame: Dengelenmiş veri seti.
        """
        if use_smote:
            smote = SMOTE(random_state=42)
            X = data.drop(columns=[class_column]).values
            y = data[class_column].values
            X_balanced, y_balanced = smote.fit_resample(X, y)

            balanced_data = pd.DataFrame(X_balanced, columns=data.columns[:-1])
            balanced_data[class_column] = y_balanced
            return balanced_data
        else:
            return data
    
    def prepare_lstm_data(self, data, use_feature_extraction=False):
        """
        Veriyi LSTM modeline uygun hale getiren metot.

        Parametreler:
            data (pd.DataFrame): Veri seti.

        Returns:
            tuple: Eğitim, doğrulama ve test için yeniden şekillendirilmiş veriler ve etiketler.
        """
        features = []
        labels = []
        unique_classes = data["class"].unique()
        for gesture in tqdm(unique_classes, desc="Sınıflar İşleniyor"):
            subset = data[data["class"] == gesture]
            for i in tqdm(range(0, len(subset) - self.window_size, self.window_size), desc=f"Sınıf {gesture} İşleniyor", leave=False):
                window = subset.iloc[i:i + self.window_size][self.channels].values
                features.append(window)
                labels.append(gesture)

        features = np.array(features)
        labels = np.array(labels)

        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_lstm_model(self, input_shape, num_classes):
        """
        LSTM modelini oluşturan metot.

        Parametreler:
            input_shape (tuple): Giriş verisinin şekli (örneğin, (window_size, num_channels)).
            num_classes (int): Sınıf sayısı.

        Returns:
            tf.keras.models.Sequential: Derlenmiş LSTM modeli.
        """
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=num_classes, activation="softmax")) # Softmax for multi-class classification

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # Use sparse_categorical_crossentropy
        return model
    
    def plot_confusion_matrix(self, y_true, y_pred, classes, filename, save_dir, scenario_name, use_filter, use_smote, data_cleaning):
        """
        Karışıklık matrisini çizen metot.
        
        Parametreler:
             y_true (np.array): Gerçek etiketler.
             y_pred (np.array): Tahmin edilen etiketler.
             classes (list): Sınıf isimleri.
             filename (str): Kaydedilecek dosya adı.
             save_dir (str): Kaydedilecek dizin.
             scenario_name (str): Senaryo adı.
             use_filter (bool): Filtreleme kullanıldı mı?
             use_smote (bool): SMOTE kullanıldı mı?
             data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)

        title = "Karışıklık Matrisi"
        feature_info = (
            f"Pencere Boyutu: {self.window_size}, "
            f"Kesim: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Örnekleme Hızı: {self.sampling_rate} Hz, "
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )

        ax.set_title(f"{title}\n{feature_info}", fontsize=10, pad=20)
        ax.set_xlabel("Tahmin Edilen", fontsize=10)
        ax.set_ylabel("Gerçek", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    def plot_loss_graph(self, train_losses, val_losses, save_dir, scenario_name, use_filter, use_smote, data_cleaning):
        """
        Kayıp grafiğini çizen metot.
        
        Parametreler:
             train_losses (list): Eğitim kayıpları.
             val_losses (list): Doğrulama kayıpları.
             save_dir (str): Kaydedilecek dizin.
             scenario_name (str): Senaryo adı.
             use_filter (bool): Filtreleme kullanıldı mı?
             use_smote (bool): SMOTE kullanıldı mı?
             data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Eğitim Kaybı")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Doğrulama Kaybı")
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp (Categorical Crossentropy)")

        feature_info = (
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )
        plt.title(f"Eğitim ve Doğrulama Kayıpları\n{feature_info}", fontsize=10)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "loss_graph.png"))
        plt.close()
        log_file = os.path.join(save_dir, "loss_plot.log")
        self.log("Kayıp grafiği 'loss_graph.png' olarak kaydedildi.", log_file)

    def train_and_evaluate_model(self, save_dir, use_filter=False, use_smote=False, model_params={}, data_cleaning=False, scenario_name=""):
        """
        Modeli eğiten ve değerlendiren ana metot.
        
        Parametreler:
            save_dir (str): Sonuçların kaydedileceği dizin.
            use_filter (bool): Filtreleme kullanılacak mı (True/False).
            use_smote (bool): SMOTE kullanılacak mı (True/False).
            model_params (dict): LSTM modelinin parametreleri.
            data_cleaning (bool): Veri temizleme kullanılacak mı (True/False).
            scenario_name (str): Senaryo adı.
        """
        start_time = time.time()  # Başlangıç zamanı
        log_file = os.path.join(save_dir, "log.txt")  # her senaryo için ayrı log dosyası
        self.log(f"Starting scenario: {scenario_name}", log_file)

        with open(log_file, "w") as f:
            f.write("Model Eğitim Logları\n")
            f.write("=" * 50 + "\n")

        data = pd.read_csv(self.file_path)
        self.log("\nVeri Yükleme Tamamlandı", log_file)
        self.log(f"Toplam Veri Sayısı: {len(data)}", log_file)

        if data_cleaning:
            data = self.drop_columns(data, ["time", "label"])
            self.log("Sütunlar Silindi: ['time', 'label']", log_file)
            data = self.drop_na(data)
            self.log("Eksik Veriler Temizlendi", log_file)
            data = self.drop_unmarked_class(data, "class", unmarked_value=0)
            self.log("Sınıf 0 Temizlendi", log_file)
            self.log(f"\nSınıf Dağılımı (Temizleme Sonrası):\n{data['class'].value_counts()}", log_file)
        else:
            self.log("Veri Temizleme Atlandı", log_file)

        data = self.filter_all_channels(data, use_filter)
        self.log(f"Bant Geçiş Filtresi Uygulandı: {self.cutoff[0]}-{self.cutoff[1]} Hz" if use_filter else "Filtreleme Atlandı", log_file)


        data = self.balance_data_with_smote(data, class_column="class", use_smote=use_smote)
        self.log(f"Sınıf Dağılımı (Dengeleme Sonrası):\n{data['class'].value_counts()}" if use_smote else "Dengeleme Atlandı", log_file)


        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_lstm_data(data)
        self.log("Eğitim, Doğrulama ve Test Setleri Ayrıldı", log_file)
        
        input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
        num_classes = len(np.unique(y_train)) #number of classes

        model = self.build_lstm_model(input_shape=input_shape, num_classes=num_classes)

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=500, batch_size=32,
                        callbacks=[early_stopping],
                        verbose=1)

        train_losses = history.history['loss']
        val_losses = history.history['val_loss']

        self.plot_loss_graph(train_losses, val_losses, save_dir, scenario_name, use_filter, use_smote, data_cleaning)

        metrics = []
        def evaluate_and_log(name, y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            metrics.append([name, accuracy, f1, precision, recall])
            self.log(f"\n{name} Metrikleri:\n"
                   f"Doğruluk: {accuracy:.4f}, F1 Skoru: {f1:.4f}, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}", log_file)
            
        y_val_pred = np.argmax(model.predict(X_val), axis=-1)
        evaluate_and_log("Doğrulama", y_val, y_val_pred)
        self.plot_confusion_matrix(y_val, y_val_pred, classes=np.unique(y_train), filename="dogrulama_karisiklik_matrisi.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, data_cleaning=data_cleaning)

        y_test_pred = np.argmax(model.predict(X_test), axis=-1)
        evaluate_and_log("Test", y_test, y_test_pred)
        self.plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(y_train), filename="test_karisiklik_matrisi.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, data_cleaning=data_cleaning)

        metrics_df = pd.DataFrame(metrics, columns=["Set", "Doğruluk", "F1 Skoru", "Kesinlik", "Duyarlılık"])
        metrics_df.to_csv(os.path.join(save_dir, "metrikler.csv"), index=False)
        self.log("Tüm metrikler 'metrikler.csv' dosyasına kaydedildi.", log_file)


        end_time = time.time()  # Bitiş zamanı
        elapsed_time = end_time - start_time
        self.log(f"Senaryo Süresi: {elapsed_time:.2f} saniye", log_file)  # Senaryo süresini logla
        self.log(f"Finished scenario: {scenario_name}", log_file)  # Senaryo bitiş mesajı

    def run_scenario(self, scenario):
        """
        Belirtilen senaryoyu çalıştıran metot.
        
        Parametreler:
            scenario (dict): Çalıştırılacak senaryo.
        """
        scenario_name = scenario["name"]
        save_dir = os.path.join(self.base_dir, scenario_name)
        self.cutoff = scenario.get("cutoff", (20,450)) # add default cutoff value if cutoff isn't there
        self.update_cutoff(self.cutoff)
        self.train_and_evaluate_model(save_dir, use_filter=scenario["use_filter"], use_smote=scenario["use_smote"],
                                      model_params=scenario["model_params"],
                                      data_cleaning=scenario["data_cleaning"], scenario_name=scenario_name)

    def run_scenarios(self):
        """
        Tüm senaryoları çalıştıran metot.
        """
        scenarios = [
             {"name": "original_data", "use_filter": False, "use_smote": False, "model_params": {}, "data_cleaning": False},
             {"name": "cleaned_data", "use_filter": False, "use_smote": False, "model_params": {}, "data_cleaning": True},
             {"name": "not_filtered_balanced_data", "use_filter": False, "use_smote": True, "model_params": {}, "data_cleaning": True},
             {"name": "filtered_data", "use_filter": True, "use_smote": False, "model_params": {}, "data_cleaning": True},
             {"name": "balanced_data", "use_filter": True, "use_smote": True, "model_params": {}, "data_cleaning": True},
             {"name": "all_processed_data_with_filtered_1_499Hz", "use_filter": True, "use_smote": True, "model_params": {}, "data_cleaning": True,"cutoff": (1, 499)}
        ]

        with multiprocessing.Pool() as pool:
            pool.map(self.run_scenario, scenarios)


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = LstmTrainer(file_path=DATASET_FILE_PATH, channels=channels, window_size=200, cutoff=(20, 450), sampling_rate=1000, patience=20, lstm_units=50, dropout_rate=0.5, learning_rate=0.001)
    trainer.run_scenarios()