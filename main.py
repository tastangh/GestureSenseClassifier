import os
import pandas as pd
import numpy as np
from enum import Enum
import time
import itertools  # Parametre kombinasyonları için

# Dataset
from dataset_filter import DatasetFilter
from dataset_cleaner import DatasetCleaner
from dataset_balancer import DatasetBalancer
from dataset_feature_extractor import DatasetFeatureExtractor
from dataset_scaler import DatasetScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from decisiontree_trainer import DecisionTreeTrainer
from lstm_trainer import LSTMTrainer
from log_reg_trainer import LogRegTrainer
from randomforest_trainer import RandomForestTrainer
from svm_trainer import SVMTrainer
from ann_trainer import ANNTrainer

# Save function
from save_results import save_results_to_excel

import tensorflow as tf

# # TensorFlow GPU memory limit configuration
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use 10GB of GPU RAM
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
#         )
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


class ModelType(Enum):
    LOGISTIC_REGRESSION = "LogisticRegression"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    LSTM = "LSTM"
    SVM = "SVM"
    # Artificial Neural Networks - Yapay Sinir Ağları
    ANN = "ANN"


def run_model(model_type, model_params, X_train, y_train, X_val, y_val, X_test, y_test, output_dir, filter_params):
    if model_type == ModelType.LOGISTIC_REGRESSION:
        print("Lojistik regresyon modeli eğitiliyor...")
        num_classes = len(np.unique(y_train))
        trainer = LogRegTrainer(input_dim=X_train.shape[1], num_classes=num_classes, output_dir=output_dir)
        trainer.train(X_train, y_train, X_val=X_val, y_val=y_val,
                      epochs=model_params.get("epochs", 10),
                      batch_size=model_params.get("batch_size", 32),
                      optimizer_type=model_params.get("optimizer_type", "adam"),
                      learning_rate=model_params.get("learning_rate", 0.001),
                      early_stopping=model_params.get("early_stopping", False),
                      patience=model_params.get("patience", 3),
                      learning_rate_scheduling=model_params.get("learning_rate_scheduling", False),
                      factor=model_params.get("factor", 0.1),
                      min_lr=model_params.get("min_lr", 1e-6)
                      )

        print("Model değerlendiriliyor...")
        y_pred = trainer.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

        save_results_to_excel(
            output_dir,
            "LogisticRegression",
            model_params,
            trainer.history['loss'][-1],  # Son epoch'un loss değeri
            trainer.history['accuracy'][-1],  # Son epoch'un accuracy değeri
            trainer.history['val_loss'][-1] if 'val_loss' in trainer.history else None,
            trainer.history['val_accuracy'][-1] if 'val_accuracy' in trainer.history else None,
             accuracy, # Test Accuracy ekledik
             filter_params, # Filtre parametreleri ekledik
             {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
              "StandardScaler", # Ölçeklendirme türü
             "SMOTE" # Dengeleme türünü ekledik.
        )
        trainer.plot_metrics()

    elif model_type == ModelType.DECISION_TREE:
        print("Decision Tree modeli eğitiliyor...")
        num_classes = len(np.unique(y_train))
        dt_trainer = DecisionTreeTrainer(input_dim=X_train.shape[1], num_classes=num_classes, max_depth=model_params["max_depth"], output_dir=output_dir)
        dt_trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)

        print("Decision Tree modeli değerlendiriliyor...")
        y_pred_dt = dt_trainer.predict(X_test)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        print(f"Decision Tree Test Doğruluğu: {accuracy_dt * 100:.2f}%")

        save_results_to_excel(output_dir, "DecisionTree", model_params, dt_trainer.history['loss'][-1] if dt_trainer.history else None,
                            dt_trainer.history['accuracy'][-1] if dt_trainer.history else None,
                            dt_trainer.history['val_loss'][-1] if dt_trainer.history and 'val_loss' in dt_trainer.history else None,
                             dt_trainer.history['val_accuracy'][-1] if dt_trainer.history and 'val_accuracy' in dt_trainer.history else None,
                              accuracy_dt, # Test Accuracy ekledik
                             filter_params, # Filtre parametreleri ekledik
                             {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
                              "StandardScaler", # Ölçeklendirme türü
                             "SMOTE" # Dengeleme türünü ekledik.
                            )
        dt_trainer.plot_metrics()

    elif model_type == ModelType.LSTM:
        print("LSTM modeli için veri hazırlanıyor...")

        time_steps = model_params["time_steps"]
        total_features = X_train.shape[1]

        # time_steps ile uyumlu num_features hesaplanıyor
        if total_features % time_steps != 0:
            raise ValueError(
                f"Özellik sayısı ({total_features}) time_steps ({time_steps}) ile uyumlu değil. time_steps değerini değiştirin.")

        num_features = total_features // time_steps

        # X_train ve X_test yeniden şekillendiriliyor
        X_train_lstm = X_train.reshape(-1, time_steps, num_features)
        X_val_lstm = X_val.reshape(-1, time_steps, num_features)
        X_test_lstm = X_test.reshape(-1, time_steps, num_features)

        print("LSTM modeli eğitiliyor...")
        num_classes = len(np.unique(y_train))
        lstm_trainer = LSTMTrainer(input_shape=(time_steps, num_features), num_classes=num_classes, lstm_units=64, output_dir=output_dir)
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

        save_results_to_excel(
            output_dir,
            "LSTM",
            model_params,
             lstm_trainer.history['loss'][-1],
            lstm_trainer.history['accuracy'][-1],
            lstm_trainer.history['val_loss'][-1] if 'val_loss' in lstm_trainer.history else None,
            lstm_trainer.history['val_accuracy'][-1] if 'val_accuracy' in lstm_trainer.history else None,
              accuracy_lstm, # Test Accuracy ekledik
             filter_params, # Filtre parametreleri ekledik
             {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
              "StandardScaler", # Ölçeklendirme türü
             "SMOTE" # Dengeleme türünü ekledik.
        )
        lstm_trainer.plot_metrics()

    elif model_type == ModelType.RANDOM_FOREST:
      print("Random Forest modeli eğitiliyor...")
      num_classes = len(np.unique(y_train))
      rf_trainer = RandomForestTrainer(input_dim=X_train.shape[1], num_classes=num_classes,
                                       n_estimators=model_params.get("n_estimators", 100),
                                        max_depth=model_params.get("max_depth", None),
                                        output_dir = output_dir)
      rf_trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)

      print("Random Forest modeli değerlendiriliyor...")
      y_pred_rf = rf_trainer.predict(X_test)
      accuracy_rf = accuracy_score(y_test, y_pred_rf)
      print(f"Random Forest Test Doğruluğu: {accuracy_rf * 100:.2f}%")

      # Sonuçları kaydet
      save_results_to_excel(
            output_dir,
            "RandomForest",
            model_params,
            rf_trainer.history['loss'][-1] if rf_trainer.history else None,
            rf_trainer.history['accuracy'][-1] if rf_trainer.history else None,
            rf_trainer.history['val_loss'][-1] if rf_trainer.history and 'val_loss' in rf_trainer.history else None,
             rf_trainer.history['val_accuracy'][-1] if rf_trainer.history and 'val_accuracy' in rf_trainer.history else None,
              accuracy_rf, # Test Accuracy ekledik
             filter_params, # Filtre parametreleri ekledik
              {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
              "StandardScaler", # Ölçeklendirme türü
             "SMOTE" # Dengeleme türünü ekledik.
        )
      rf_trainer.plot_metrics()


    elif model_type == ModelType.SVM:
        print("SVM modeli eğitiliyor...")
        num_classes = len(np.unique(y_train))
        svm_trainer = SVMTrainer(input_dim=X_train.shape[1], num_classes = num_classes,
            C=model_params.get("C", 1.0), output_dir=output_dir
        )
        svm_trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)

        print("SVM modeli değerlendiriliyor...")
        y_pred_svm = svm_trainer.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        print(f"SVM Test Doğruluğu: {accuracy_svm * 100:.2f}%")

        # Sonuçları kaydet
        save_results_to_excel(
            output_dir,
            "SVM",
            model_params,
            svm_trainer.history['loss'][-1] if svm_trainer.history else None,
            svm_trainer.history['accuracy'][-1] if svm_trainer.history else None,
            svm_trainer.history['val_loss'][-1] if svm_trainer.history and 'val_loss' in svm_trainer.history else None,
            svm_trainer.history['val_accuracy'][-1] if svm_trainer.history and 'val_accuracy' in svm_trainer.history else None,
             accuracy_svm, # Test Accuracy ekledik
             filter_params, # Filtre parametreleri ekledik
             {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
              "StandardScaler", # Ölçeklendirme türü
             "SMOTE" # Dengeleme türünü ekledik.
        )
        svm_trainer.plot_metrics()

    elif model_type == ModelType.ANN:
        print("Yapay Sinir Ağları (ANN) modeli eğitiliyor...")
        num_classes = len(np.unique(y_train))
        ann_trainer = ANNTrainer(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_layers=model_params.get("hidden_layers", [64, 32]),
            dropout_rate=model_params.get("dropout_rate", 0.2),
            learning_rate=model_params.get("learning_rate", 0.001),
            output_dir=output_dir
        )
        ann_trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=model_params.get("epochs", 10),
            batch_size=model_params.get("batch_size", 32),
            early_stopping=model_params.get("early_stopping", False),
            patience=model_params.get("patience", 3),
            learning_rate_scheduling=model_params.get("learning_rate_scheduling", False),
            factor=model_params.get("factor", 0.1),
            min_lr=model_params.get("min_lr", 1e-6)
        )

        print("ANN modeli değerlendiriliyor...")
        y_pred_ann = ann_trainer.predict(X_test)
        accuracy_ann = accuracy_score(y_test, y_pred_ann)
        print(f"ANN Test Doğruluğu: {accuracy_ann * 100:.2f}%")

        # Sonuçları kaydet
        save_results_to_excel(
            output_dir,
            "ANN",
            model_params,
             ann_trainer.history['loss'][-1],
            ann_trainer.history['accuracy'][-1],
            ann_trainer.history['val_loss'][-1] if 'val_loss' in ann_trainer.history else None,
            ann_trainer.history['val_accuracy'][-1] if 'val_accuracy' in ann_trainer.history else None,
             accuracy_ann, # Test Accuracy ekledik
             filter_params, # Filtre parametreleri ekledik
              {"window_size": 200}, # Öznitelik çıkarma parametrelerini ekledik.
              "StandardScaler", # Ölçeklendirme türü
             "SMOTE" # Dengeleme türünü ekledik.
        )
        ann_trainer.plot_metrics()

    else:
        print("Geçersiz model türü seçildi!")


def test_filter_settings(data, channels, sampling_rate, filter_type, cutoff, order, apply_notch, notch_freq, show_plots=True, output_dir="./output"):

    filter_processor = DatasetFilter(data, channels, sampling_rate)
    filter_processor.filter_all_channels(filter_type=filter_type, cutoff=cutoff, order=order, apply_notch=apply_notch, notch_freq=notch_freq)
    filtered_data = filter_processor.get_filtered_data()

    if show_plots:
        frequency_plot_path = os.path.join(output_dir, f"frequency_spectra_{filter_type}.png")
        filter_processor.plot_frequency_spectrum(
            signals=[data[channel] for channel in channels],
            filtered_signals=[filtered_data[channel] for channel in channels],
            titles=[f"{channel} - Frekans Spektrumu" for channel in channels],
            output_path=frequency_plot_path
        )

        time_plot_path = os.path.join(output_dir, f"time_domain_signals_{filter_type}.png")
        filter_processor.plot_signals(
            signals=[data[channel] for channel in channels],
            filtered_signals=[filtered_data[channel] for channel in channels],
            titles=[f"{channel} - Zaman Domeni Sinyalleri" for channel in channels],
            output_path=time_plot_path,
            start=0,
            end=1000
        )
    return filtered_data


def main(file_path, selected_models, filter_params):
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu.")

    # 1. Veri Yükleme
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)
    channels = [f"channel{i}" for i in range(1, 9)]
    
    print("Veri Yükleme sonrası sınıf dağılımı:")
    print(data['class'].value_counts())
    
    # 2. Veri Temizleme: Sınıfları ve sütunları kaldır
    print("Veri temizleme işlemi yapılıyor...")
    cleaner = DatasetCleaner()
    
    # Sınıfları sil
    data = cleaner.drop_classes(data, [0, 7])
    print("Sınıf silme sonrası sınıf dağılımı:")
    print(data['class'].value_counts())
    
    # Gereksiz sütunları sil
    data = cleaner.drop_columns(data, columns=["label"])
    print("Sütun silme sonrası sınıf dağılımı:")
    print(data['class'].value_counts())

    # 3. Veri Dengeleme
    print("Veri SMOTE ile dengeleniyor...")
    features, labels = DatasetFeatureExtractor.extract_features(data, channels, window_size=2)
    balancer = DatasetBalancer()
    features, labels = balancer.balance(features, labels)
    print("Dengeleme sonrası sınıf dağılımı:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Sınıf {label}: {count} örnek")
        
    # 4. Filtreleme
    print("Tüm kanallar için filtreleme işlemi yapılıyor...")
    filtered_data = test_filter_settings(data, channels, sampling_rate=1000,
                                        filter_type=filter_params["filter_type"],
                                        cutoff=filter_params["cutoff"],
                                        order=filter_params["order"],
                                        apply_notch=filter_params["apply_notch"],
                                        notch_freq=filter_params["notch_freq"],
                                        show_plots=filter_params["show_plots"],
                                        output_dir=output_dir)
    filtered_data['class'] = data['class']
    print("Filtreleme sonrası sınıf dağılımı:")
    print(filtered_data['class'].value_counts())

    # 5. Özellik Çıkarma
    print("Özellikler çıkarılıyor...")
    features, labels = DatasetFeatureExtractor.extract_features(filtered_data, channels, window_size=2)
    if np.isnan(features).any():
        print("Error: Features contain NaN values.")
    if np.isinf(features).any():
        print("Error: Features contain infinite values.")
    
    print("Özellik çıkarma sonrası sınıf dağılımı:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Sınıf {label}: {count} örnek")
    
    # 6. Veri Ölçekleme ve Train/Validation/Test Split
    print("Veri ölçekleniyor ve eğitim/test ayrımı yapılıyor...")
    scaler = DatasetScaler()
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("Veri ölçekleme sonrası sınıf dağılımları:")
    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    print("Eğitim seti dağılımı:")
    for label, count in zip(unique_train_labels, train_counts):
         print(f"Sınıf {label}: {count} örnek")
    
    unique_val_labels, val_counts = np.unique(y_val, return_counts=True)
    print("Validation seti dağılımı:")
    for label, count in zip(unique_val_labels, val_counts):
         print(f"Sınıf {label}: {count} örnek")
    
    unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
    print("Test seti dağılımı:")
    for label, count in zip(unique_test_labels, test_counts):
          print(f"Sınıf {label}: {count} örnek")

    # Model Eğitimi ve Değerlendirmesi
    for model_type, model_params in selected_models:
        run_model(model_type, model_params, X_train, y_train, X_val, y_val, X_test, y_test, output_dir=output_dir, filter_params=filter_params)

    # Sonuçların kaydedilmesi
    print(f"Model eğitimi ve değerlendirme başarıyla tamamlandı. Sonuçlar '{output_dir}' klasörüne kaydedildi.")


if __name__ == "__main__":
    dataset_path = "dataset/EMG-data.csv"

    # Filtre parametreleri
    filter_params = {
        "filter_type": "band",
        "cutoff": (1, 499),
        "order": 4,
        "apply_notch": False,
        "notch_freq": 50,
        "show_plots": False
    }
    
    # Model parametreleri ve hangi modelin çalıştırılacağını belirleme
    selected_models = [
        (ModelType.LOGISTIC_REGRESSION, {
            "learning_rate": [0.001, 0.0001],
            "epochs": [10, 20],
            "batch_size": [32, 64],
            "optimizer_type": ["adam", "sgd"],
            "early_stopping": [True, False],
            "patience": [5, 10],
            "learning_rate_scheduling": [True, False],
            "factor": [0.1, 0.2],
            "min_lr": [1e-6, 1e-5]
        }),
         (ModelType.DECISION_TREE, { "max_depth": [20, 30]}), # Decision Tree
        (ModelType.RANDOM_FOREST, { "n_estimators": [100, 150],  "max_depth": [10, 20], "random_state": [42]}), # Random Forest
         (ModelType.ANN, {"hidden_layers": [[32], [64, 32]], "dropout_rate": [0.2, 0.3], "learning_rate": [0.01, 0.001], "epochs": [10, 20], "batch_size": [32, 64] ,
                          "early_stopping": [True, False],
                          "patience": [5, 10],
                          "learning_rate_scheduling": [True, False],
                          "factor": [0.1, 0.2],
                          "min_lr": [1e-6, 1e-5]}), # ANN
         (ModelType.LSTM, { "time_steps":[8] , "lstm_units": [64, 128],"epochs": [10, 20], "batch_size": [64, 90] ,
                           "learning_rate": [0.001, 0.0005],
                           "early_stopping": [True, False],
                           "patience": [5, 10],
                           "learning_rate_scheduling": [True, False],
                           "factor": [0.1, 0.2],
                           "min_lr": [1e-6, 1e-5]}), # LSTM
        (ModelType.SVM, {"C": [0.1, 1.0], "random_state": [42]}) # SVM
    ]
    
    
    all_model_params = []
    for model_type, params in selected_models:
          keys, values = zip(*params.items())
          all_params_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
          all_model_params.extend([(model_type, param_set) for param_set in all_params_combinations])
          
    baslangic = time.time()
    for model_type, model_params in all_model_params:
       main(dataset_path, selected_models=[(model_type, model_params)], filter_params=filter_params)
    bitis = time.time()

    print("Toplam geçen süre :", bitis-baslangic)