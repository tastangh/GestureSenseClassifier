import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import log_loss
import warnings
import time
import multiprocessing

warnings.filterwarnings("ignore")  # Uyarıları kapat

class DTCTrainer:
    def __init__(self, file_path, channels, window_size=200, cutoff=(20, 450), sampling_rate=1000, base_dir="results"):
        self.file_path = file_path
        self.channels = channels
        self.window_size = window_size
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.base_dir = base_dir

    def log(self, message, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    @staticmethod
    def drop_columns(data, columns):
        return data.drop(columns=columns, errors="ignore")

    @staticmethod
    def drop_unmarked_class(data, class_column, unmarked_value=0):
        return data[data[class_column] != unmarked_value].reset_index(drop=True)

    @staticmethod
    def drop_na(data):
        return data.dropna().reset_index(drop=True)

    @staticmethod
    def apply_filter(signal, filter_type, cutoff, order=4, sampling_rate=1000):
        nyquist = 0.5 * sampling_rate
        if filter_type == "band":
            normalized_cutoff = [freq / nyquist for freq in cutoff]
            b, a = butter(order, normalized_cutoff, btype="band")
        else:
            raise ValueError("Sadece bant geçiş filtresi destekleniyor.")
        return filtfilt(b, a, signal)

    def filter_all_channels(self, data, use_filter=True):
        if use_filter:
            for channel in self.channels:
                data[channel] = self.apply_filter(data[channel].values, "band", self.cutoff, order=4, sampling_rate=self.sampling_rate)
        return data

    @staticmethod
    def balance_data_with_smote(data, class_column, use_smote=True, balance_technique="smote"):
        if use_smote:
            if balance_technique == "smote":
                smote = SMOTE(random_state=42)
                X = data.drop(columns=[class_column]).values
                y = data[class_column].values
                X_balanced, y_balanced = smote.fit_resample(X, y)

                balanced_data = pd.DataFrame(X_balanced, columns=data.columns[:-1])
                balanced_data[class_column] = y_balanced
                return balanced_data
            elif balance_technique == "random":
                ros = RandomOverSampler(random_state=42)
                X = data.drop(columns=[class_column]).values
                y = data[class_column].values
                X_balanced, y_balanced = ros.fit_resample(X, y)
                balanced_data = pd.DataFrame(X_balanced, columns=data.columns[:-1])
                balanced_data[class_column] = y_balanced
                return balanced_data
        else:
            return data

    @staticmethod
    def advanced_feature_extraction(window, feature_types=["all"]):
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
        features = []
        labels = []
        unique_classes = data["class"].unique()
        for gesture in tqdm(unique_classes, desc="Sınıflar İşleniyor"):
            subset = data[data["class"] == gesture]
            for i in tqdm(range(0, len(subset) - self.window_size, self.window_size), desc=f"Sınıf {gesture} İşleniyor", leave=False):
                window = subset.iloc[i:i + self.window_size][self.channels].values
                extracted_features = self.advanced_feature_extraction(window, feature_types=feature_types)
                features.append(extracted_features)
                labels.append(gesture)
        return np.array(features), np.array(labels)

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename, save_dir, scenario_name, use_filter, use_smote, use_feature_extraction, data_cleaning):
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

    def plot_tree_graph(self, model, save_dir, scenario_name, feature_names=None):
        plt.figure(figsize=(20,15))
        plot_tree(model, filled=True, feature_names=feature_names, fontsize=8)
        plt.title(f"Decision Tree (Senaryo: {scenario_name})", fontsize=10)
        plt.savefig(os.path.join(save_dir, "decision_tree.png"))
        plt.close()

    def train_and_evaluate_model(self, save_dir, use_filter=False, use_smote=False, feature_types=["all"], model_params={}, data_cleaning=False, normalization=True, use_feature_extraction=False, scenario_name="", balance_technique="smote", data_ratio=1):
        start_time = time.time()
        log_file = os.path.join(save_dir, "log.txt")
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

        data = self.balance_data_with_smote(data, class_column="class", use_smote=use_smote, balance_technique=balance_technique)
        self.log(f"Sınıf Dağılımı (Dengeleme Sonrası):\n{data['class'].value_counts()}" if use_smote else "Dengeleme Atlandı", log_file)

        features, labels = None, None
        if use_feature_extraction:
            features, labels = self.extract_features(data, feature_types=feature_types)
            self.log(f"\nÖzellik Çıkarımı Tamamlandı: {len(features)} örnek", log_file)
            if feature_types == ["all"]:
                feature_names = ["mean_abs_value", "root_mean_square", "waveform_length", "variance", "integrated_emg", "mean_frequency"]
                feature_names = [f"{feature}_channel{channel}" for feature in feature_names for channel in range(1, 9)]
            else:
                feature_names =  [f"{feature_type}_channel{channel}" for feature_type in feature_types  for channel in range(1, 9)]
        else:
            features = data[self.channels].values
            labels = data["class"].values
            self.log("Özellik Çıkarımı Atlandı", log_file)
            feature_names = self.channels
        
        if data_ratio < 1 :
            X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=1 - data_ratio, random_state=42, stratify=labels)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            self.log(f"Veri Seti Oranı {data_ratio} olarak güncellendi ve test verisi ayrıldı", log_file)
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            self.log(f"Veri Seti Oranı {data_ratio}, standart test verisi ayrıldı", log_file)

        self.log("Eğitim, Doğrulama ve Test Setleri Ayrıldı", log_file)

        scaler = StandardScaler()
        if normalization:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            self.log("Veri Normalizasyonu Tamamlandı", log_file)
        else:
            self.log("Veri Normalizasyonu Atlandı", log_file)

        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 10],
            "criterion": ["gini", "entropy"]
        }
        
        best_score = 0
        best_params = {}
        best_model = None
        
        patience = 3
        best_val_score = 0
        no_improve_count = 0
        
        for depth in param_grid["max_depth"]:
            for min_split in param_grid["min_samples_split"]:
              for min_leaf in param_grid["min_samples_leaf"]:
                for criterion in param_grid["criterion"]:
                    
                    model = DecisionTreeClassifier(random_state=42, max_depth = depth, min_samples_split = min_split, min_samples_leaf = min_leaf, criterion = criterion )
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    
                    if score > best_val_score:
                        best_val_score = score
                        best_params = {"max_depth": depth, "min_samples_split": min_split, "min_samples_leaf": min_leaf, "criterion": criterion}
                        best_model = model
                        no_improve_count = 0
                        
                        
                    else:
                       no_improve_count +=1
                       
                    if no_improve_count >= patience:
                      self.log(f"Eğitim erken durduruldu, iterasyon {depth}_{min_split}_{min_leaf}_{criterion} de en iyi score elde edildi", log_file)
                      break
                if no_improve_count >= patience:
                  break
              if no_improve_count >= patience:
                break
        
        self.log(f"En İyi Parametreler: {best_params}", log_file)

        metrics = []
        def evaluate_and_log(name, y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            metrics.append([name, accuracy, f1, precision, recall])
            self.log(f"\n{name} Metrikleri:\n"
                    f"Doğruluk: {accuracy:.4f}, F1 Skoru: {f1:.4f}, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}", log_file)

        y_val_pred = best_model.predict(X_val)
        evaluate_and_log("Doğrulama", y_val, y_val_pred)
        self.plot_confusion_matrix(y_val, y_val_pred, classes=np.unique(labels), filename="dogrulama_karisiklik_matrisi.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)

        y_test_pred = best_model.predict(X_test)
        evaluate_and_log("Test", y_test, y_test_pred)
        self.plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(labels), filename="test_karisiklik_matrisi.png", save_dir=save_dir, scenario_name=scenario_name, use_filter=use_filter, use_smote=use_smote, use_feature_extraction=use_feature_extraction, data_cleaning=data_cleaning)

        metrics_df = pd.DataFrame(metrics, columns=["Set", "Doğruluk", "F1 Skoru", "Kesinlik", "Duyarlılık"])
        metrics_df.to_csv(os.path.join(save_dir, "metrikler.csv"), index=False)
        self.log("Tüm metrikler 'metrikler.csv' dosyasına kaydedildi.", log_file)

        self.plot_tree_graph(best_model, save_dir, scenario_name, feature_names=feature_names)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.log(f"Senaryo Süresi: {elapsed_time:.2f} saniye", log_file)
        self.log(f"Finished scenario: {scenario_name}", log_file)

    def run_scenario(self, scenario):
        scenario_name = scenario["name"]
        save_dir = os.path.join(self.base_dir, scenario_name)
        self.train_and_evaluate_model(save_dir, use_filter=scenario["use_filter"], use_smote=scenario["use_smote"],
                                      feature_types=scenario["feature_types"],
                                      data_cleaning=scenario["data_cleaning"], normalization=scenario["normalization"],
                                      use_feature_extraction=scenario["use_feature_extraction"], scenario_name=scenario_name, balance_technique=scenario.get("balance_technique", "smote"), data_ratio=scenario.get("data_ratio", 1))

    def run_scenarios(self):
        scenarios = [
            {"name": "original_data_with_default_params", "use_filter": False, "use_smote": False, "feature_types": ["all"], "data_cleaning": False, "normalization": True, "use_feature_extraction": False},
            {"name": "compare_gini_entropy", "use_filter": True, "use_smote": False, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "model_params": {"criterion": "gini"}},
            {"name": "compare_gini_entropy_with_entropy", "use_filter": True, "use_smote": False, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "model_params": {"criterion": "entropy"}},
            {"name": "time_domain_features", "use_filter": True, "use_smote": False, "feature_types": ["mean_abs_value", "root_mean_square", "waveform_length", "variance", "integrated_emg"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True},
            {"name": "frequency_domain_features", "use_filter": True, "use_smote": False, "feature_types": ["mean_frequency"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True},
            {"name": "all_features_together", "use_filter": True, "use_smote": False, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True},
            {"name": "full_dataset_training", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True},
           {"name": "data_ratio_20", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "data_ratio": 0.2},
           {"name": "data_ratio_40", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "data_ratio": 0.4},
           {"name": "data_ratio_60", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "data_ratio": 0.6},
           {"name": "data_ratio_80", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "data_ratio": 0.8},
           {"name": "with_smote_data_balancing", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "balance_technique" : "smote"},
            {"name": "with_randomoversampler_data_balancing", "use_filter": True, "use_smote": True, "feature_types": ["all"], "data_cleaning": True, "normalization": True, "use_feature_extraction": True, "balance_technique" : "random"},
        ]

        with multiprocessing.Pool() as pool:
            pool.map(self.run_scenario, scenarios)

if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = DTCTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=100, cutoff=(1, 499), sampling_rate=1000)
    trainer.run_scenarios()