import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
from matplotlib.colors import ListedColormap


warnings.filterwarnings("ignore")

class RandomForestTrainer:
    """
    Random Forest modelini eğitmek, değerlendirmek ve görselleştirmek için kullanılan sınıf.
    """

    def __init__(self, file_path, channels, window_size=100, cutoff=(1, 499), sampling_rate=1000, base_dir="results"):
      """
        Sınıfın başlatıcı metodu.
        Parametreler:
            file_path (str): Veri setinin dosya yolu.
            channels (list): Kullanılacak kanal isimlerinin listesi.
            window_size (int): Özellik çıkarımı için pencere boyutu.
            cutoff (tuple): Filtreleme için kesim frekansları (alt, üst).
            sampling_rate (int): Örnekleme frekansı.
            base_dir (str): Sonuçların kaydedileceği temel dizin.
      """
      self.file_path = file_path
      self.channels = channels
      self.window_size = window_size
      self.cutoff = cutoff
      self.sampling_rate = sampling_rate
      self.base_dir = base_dir
    
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
                data[channel] = self.apply_filter(data[channel].values, "band", self.cutoff, order=4,
                                                   sampling_rate=self.sampling_rate)
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

    @staticmethod
    def advanced_feature_extraction(window, feature_types=["all"]):
        """
        Pencereden gelişmiş özellikler çıkaran statik metot.
        Parametreler:
            window (np.array): Pencere verisi.
            feature_types (list): Çıkarılacak özelliklerin listesi.
        Returns:
            np.array: Çıkarılmış özellikler.
        """
        all_features = []

        if "mean_abs_value" in feature_types or "all" in feature_types:
            mean_abs_value = np.mean(np.abs(window), axis=0)
            all_features.append(mean_abs_value)

        if "root_mean_square" in feature_types or "all" in feature_types:
            root_mean_square = np.sqrt(np.mean(np.square(window), axis=0))
            all_features.append(root_mean_square)

        if "waveform_length" in feature_types or "all" in feature_types:
            waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
            all_features.append(waveform_length)

        if "variance" in feature_types or "all" in feature_types:
            variance = np.var(window, axis=0)
            all_features.append(variance)

        if "integrated_emg" in feature_types or "all" in feature_types:
            integrated_emg = np.sum(np.abs(window), axis=0)
            all_features.append(integrated_emg)

        if "mean_frequency" in feature_types or "all" in feature_types:
            freq_spectrum = np.fft.fft(window, axis=0)
            power_spectral_density = np.abs(freq_spectrum) ** 2
            mean_frequency = np.array([
                np.sum(np.arange(len(psd)) * psd) / np.sum(psd)
                for psd in power_spectral_density.T
            ])
            all_features.append(mean_frequency)

        return np.concatenate(all_features)

    def extract_features(self, data, feature_types=["all"]):
        """
        Veri setinden özellikler çıkaran metot.
        Parametreler:
            data (pd.DataFrame): Veri seti.
            feature_types (list): Çıkarılacak özelliklerin listesi.
        Returns:
            tuple: Çıkarılmış özellikler ve etiketler.
        """
        features = []
        labels = []
        unique_classes = data["class"].unique()
        for gesture in tqdm(unique_classes, desc="Sınıflar İşleniyor"):
            subset = data[data["class"] == gesture]
            for i in tqdm(range(0, len(subset) - self.window_size, self.window_size), desc=f"Sınıf {gesture} İşleniyor",
                          leave=False):
                window = subset.iloc[i:i + self.window_size][self.channels].values
                extracted_features = self.advanced_feature_extraction(window, feature_types=feature_types)
                features.append(extracted_features)
                labels.append(gesture)
        return np.array(features), np.array(labels)


    def plot_confusion_matrix(self, y_true, y_pred, classes, filename, save_dir, scenario_name, use_filter, use_smote,
                              use_feature_extraction, data_cleaning):
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
             use_feature_extraction (bool): Özellik çıkarımı kullanıldı mı?
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
            f"Özellik Çıkarımı: {'Uygulandı' if use_feature_extraction else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )

        ax.set_title(f"{title}\n{feature_info}", fontsize=10, pad=20)
        ax.set_xlabel("Tahmin Edilen", fontsize=10)
        ax.set_ylabel("Gerçek", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    def plot_loss_graph(self, train_losses, val_losses, save_dir, scenario_name, use_filter, use_smote,
                        use_feature_extraction, data_cleaning):
        """
        Kayıp grafiğini çizen metot.
        Parametreler:
             train_losses (list): Eğitim kayıpları.
             val_losses (list): Doğrulama kayıpları.
             save_dir (str): Kaydedilecek dizin.
             scenario_name (str): Senaryo adı.
             use_filter (bool): Filtreleme kullanıldı mı?
             use_smote (bool): SMOTE kullanıldı mı?
             use_feature_extraction (bool): Özellik çıkarımı kullanıldı mı?
             data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Eğitim Kaybı")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Doğrulama Kaybı")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")

        feature_info = (
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Özellik Çıkarımı: {'Uygulandı' if use_feature_extraction else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )
        plt.title(f"Eğitim ve Doğrulama Kayıpları\n{feature_info}", fontsize=10)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "loss_graph.png"))
        plt.close()
        log_file = os.path.join(save_dir, "loss_plot.log")
        self.log("Kayıp grafiği 'loss_graph.png' olarak kaydedildi.", log_file)

    def plot_feature_importance(self, importances, feature_names, filename, save_dir, scenario_name, use_filter, use_smote, use_feature_extraction, data_cleaning):
      """
        Özellik önemini çizen metot.
        Parametreler:
            importances (np.array): Özellik önem dereceleri.
            feature_names (list): Özellik isimleri.
            filename (str): Kaydedilecek dosya adı.
            save_dir (str): Kaydedilecek dizin.
            scenario_name (str): Senaryo adı.
            use_filter (bool): Filtreleme kullanıldı mı?
            use_smote (bool): SMOTE kullanıldı mı?
            use_feature_extraction (bool): Özellik çıkarımı kullanıldı mı?
            data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
      plt.figure(figsize=(10, 6))
      sorted_indices = np.argsort(importances)[::-1]  # En önemliden en aza sırala
      sorted_importances = importances[sorted_indices]
      sorted_feature_names = [feature_names[i] for i in sorted_indices]
      plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
      plt.xticks(rotation=90)

      feature_info = (
            f"Pencere Boyutu: {self.window_size}, "
            f"Kesim: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Örnekleme Hızı: {self.sampling_rate} Hz, "
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Özellik Çıkarımı: {'Uygulandı' if use_feature_extraction else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )

      plt.title(f"Özellik Önemi\n{feature_info}", fontsize=10, pad=20)
      plt.xlabel("Özellikler")
      plt.ylabel("Önem Derecesi")
      plt.tight_layout()
      plt.savefig(os.path.join(save_dir, filename))
      plt.close()


    def plot_roc_curve(self, y_true, y_pred_prob, filename, save_dir, scenario_name, use_filter, use_smote, use_feature_extraction, data_cleaning):
        """
        ROC eğrisini çizen metot.
        Parametreler:
            y_true (np.array): Gerçek etiketler.
            y_pred_prob (np.array): Tahmin edilen olasılıklar.
            filename (str): Kaydedilecek dosya adı.
            save_dir (str): Kaydedilecek dizin.
             scenario_name (str): Senaryo adı.
             use_filter (bool): Filtreleme kullanıldı mı?
             use_smote (bool): SMOTE kullanıldı mı?
             use_feature_extraction (bool): Özellik çıkarımı kullanıldı mı?
             data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        feature_info = (
            f"Pencere Boyutu: {self.window_size}, "
            f"Kesim: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Örnekleme Hızı: {self.sampling_rate} Hz, "
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Özellik Çıkarımı: {'Uygulandı' if use_feature_extraction else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )
        plt.title(f"ROC Eğrisi\n{feature_info}", fontsize=10, pad=20)
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_prob, filename, save_dir, scenario_name, use_filter, use_smote, use_feature_extraction, data_cleaning):
        """
        Precision-Recall eğrisini çizen metot.
        Parametreler:
            y_true (np.array): Gerçek etiketler.
            y_pred_prob (np.array): Tahmin edilen olasılıklar.
            filename (str): Kaydedilecek dosya adı.
            save_dir (str): Kaydedilecek dizin.
            scenario_name (str): Senaryo adı.
            use_filter (bool): Filtreleme kullanıldı mı?
            use_smote (bool): SMOTE kullanıldı mı?
            use_feature_extraction (bool): Özellik çıkarımı kullanıldı mı?
            data_cleaning (bool): Veri temizleme kullanıldı mı?
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Eğrisi (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        feature_info = (
            f"Pencere Boyutu: {self.window_size}, "
            f"Kesim: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Örnekleme Hızı: {self.sampling_rate} Hz, "
            f"Senaryo: {scenario_name}, "
            f"Filtreleme: {'Uygulandı' if use_filter else 'Uygulanmadı'}, "
            f"Dengeleme: {'Uygulandı' if use_smote else 'Uygulanmadı'}, "
            f"Özellik Çıkarımı: {'Uygulandı' if use_feature_extraction else 'Uygulanmadı'}, "
            f"Veri Temizleme: {'Uygulandı' if data_cleaning else 'Uygulanmadı'}"
        )

        plt.title(f"Precision-Recall Eğrisi\n{feature_info}", fontsize=10, pad=20)
        plt.legend(loc="lower left")
        plt.grid()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    def train_and_evaluate_model(self, save_dir, use_filter=False, use_smote=False, feature_types=["all"],
                                 model_params={}, data_cleaning=False, normalization=True,
                                 use_feature_extraction=False, scenario_name=""):
        """
        Modeli eğiten ve değerlendiren ana metot.
        Parametreler:
            save_dir (str): Sonuçların kaydedileceği dizin.
            use_filter (bool): Filtreleme kullanılacak mı (True/False).
            use_smote (bool): SMOTE kullanılacak mı (True/False).
            feature_types (list): Kullanılacak özelliklerin listesi.
            model_params (dict): Random Forest modelinin parametreleri.
            data_cleaning (bool): Veri temizleme kullanılacak mı (True/False).
            normalization (bool): Normalizasyon kullanılacak mı (True/False).
            use_feature_extraction (bool): Özellik çıkarımı kullanılacak mı (True/False).
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

        features, labels = None, None
        if use_feature_extraction:
            features, labels = self.extract_features(data, feature_types=feature_types)
            self.log(f"\nÖzellik Çıkarımı Tamamlandı: {len(features)} örnek", log_file)
        else:
            features = data[self.channels].values
            labels = data["class"].values
            self.log("Özellik Çıkarımı Atlandı", log_file)

        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        self.log("Eğitim, Doğrulama ve Test Setleri Ayrıldı", log_file)

        scaler = StandardScaler()
        if normalization:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            self.log("Veri Normalizasyonu Tamamlandı", log_file)
        else:
            self.log("Veri Normalizasyonu Atlandı", log_file)

        # Grid Search ile en iyi parametreleri bul
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }

        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=0, cv=3, scoring="f1_macro")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        self.log(f"Grid Search Sonuçları: En İyi Parametreler: {best_params}", log_file)
        
        feature_names = [f"feature_{i}" for i in range(features.shape[1])] # Özellik isimleri

        # Random Forest modelini eğit
        best_model.fit(X_train, y_train)
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)

        # Log loss hesaplamaları (Random Forest olasılık çıktısı var)
        train_loss = log_loss(y_train, best_model.predict_proba(X_train))
        val_loss = log_loss(y_val, best_model.predict_proba(X_val))
        
        self.log(f"Eğitim Kaybı: {train_loss:.4f}, Doğrulama Kaybı: {val_loss:.4f}", log_file)
        self.plot_loss_graph([train_loss], [val_loss], save_dir, scenario_name, use_filter, use_smote, use_feature_extraction, data_cleaning)

        metrics = []
        def evaluate_and_log(name, y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            metrics.append([name, accuracy, f1, precision, recall])
            self.log(f"\n{name} Metrikleri:\n"
                     f"Doğruluk: {accuracy:.4f}, F1 Skoru: {f1:.4f}, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}",
                     log_file)
            return accuracy, f1, precision, recall
        
        y_val_pred_prob = best_model.predict_proba(X_val)
        y_test_pred_prob = best_model.predict_proba(X_test)
        
        val_accuracy, val_f1, val_precision, val_recall = evaluate_and_log("Doğrulama", y_val, y_val_pred)
        self.plot_confusion_matrix(y_val, y_val_pred, classes=np.unique(labels),
                                   filename="dogrulama_karisiklik_matrisi.png", save_dir=save_dir,
                                   scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote,
                                   use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)
        self.plot_roc_curve(y_val, y_val_pred_prob, filename="dogrulama_roc_curve.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)
        self.plot_precision_recall_curve(y_val, y_val_pred_prob, filename="dogrulama_pr_curve.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)

        test_accuracy, test_f1, test_precision, test_recall = evaluate_and_log("Test", y_test, y_test_pred)
        self.plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(labels),
                                   filename="test_karisiklik_matrisi.png", save_dir=save_dir,
                                   scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote,
                                   use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)
        self.plot_roc_curve(y_test, y_test_pred_prob, filename="test_roc_curve.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)
        self.plot_precision_recall_curve(y_test, y_test_pred_prob, filename="test_pr_curve.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)

        # Özellik önemini çiz
        self.plot_feature_importance(best_model.feature_importances_, feature_names, filename="ozellik_onemi.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)


        metrics_df =        metrics_df = pd.DataFrame({
            'Set': ["Doğrulama", "Test"],
            'Doğruluk': [val_accuracy, test_accuracy],
            'F1 Skoru': [val_f1, test_f1],
            'Kesinlik': [val_precision, test_precision],
            'Duyarlılık': [val_recall, test_recall]
        })
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
        self.cutoff = scenario.get("cutoff", (1, 499))  # add default cutoff value if cutoff isn't there
        self.window_size = scenario.get("window_size", 100) # pencere boyutunu al
        self.train_and_evaluate_model(save_dir, use_filter=scenario["use_filter"], use_smote=scenario["use_smote"],
                                      feature_types=scenario["feature_types"], model_params=scenario["model_params"],
                                      data_cleaning=scenario["data_cleaning"], normalization=scenario["normalization"],
                                      use_feature_extraction=scenario["use_feature_extraction"],
                                      scenario_name=scenario_name)

    def run_scenarios(self):
        """
        Random Forest için senaryo tanımlamaları.
        """
        scenarios = []

        # 1. Ham Veri ile Performans
        scenarios.append({
            "name": "rf_raw_data",
            "use_filter": False,
            "use_smote": False,
            "feature_types": ["all"],
            "model_params": {},
            "data_cleaning": False,
            "normalization": True,
            "use_feature_extraction": False,
            "cutoff": (1, 499),
            "window_size": 100
        })

        # 2. en iyi parametreler 2.5 milyonda augmentaiton
        scenarios.append({
            "name": "rf_best_params_cleaning_False",
            "use_filter": True,
            "use_smote": True,
            "feature_types": ["all"],
            "model_params": {},
            "data_cleaning": False,
            "normalization": True,
            "use_feature_extraction": False,
            "cutoff": (1, 499),
            "window_size": 100
        })

        # 3. en iyi parametreler 250000'de clas 0 sız augmentaiton 
        scenarios.append({
            "name": "rf_best_params_cleaning_True",
            "use_filter": True,
            "use_smote": True,
            "feature_types": ["all"],
            "model_params": {},
            "data_cleaning": True,
            "normalization": True,
            "use_feature_extraction": False,
            "cutoff": (1, 499),
            "window_size": 100
        })

        from multiprocessing import cpu_count

        # Maksimum worker sayısını belirleyin
        MAX_WORKERS = max(1, cpu_count() - 1)  # Çekirdek sayısının 1 eksiği kadar kullanın.

        # Paralel işlem için senaryoları çalıştırma
        with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
            pool.map(self.run_scenario, scenarios)


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = RandomForestTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=100, cutoff=(1, 499),
                           sampling_rate=1000)
    trainer.run_scenarios()