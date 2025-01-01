import os
import pandas as pd
import numpy as np
from enum import Enum
import time

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

class ModelType(Enum):
    
    LOGISTIC_REGRESSION = "LogisticRegression"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    LSTM = "LSTM"
    SVM = "SVM"
    
    # Artificial Neural Networks - Yapay Sinir Ağları
    ANN = "ANN"
    
def run_model(model_type, model_params, X_train, y_train, X_test, y_test, output_dir):
    
    if model_type == ModelType.LOGISTIC_REGRESSION:
        print("Lojistik regresyon modeli eğitiliyor...")
        trainer = LogRegTrainer(model_params["max_iter"])
        trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        print("Model değerlendiriliyor...")
        y_pred = trainer.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Doğruluğu: {accuracy * 100:.2f}%")
        
        save_results_to_excel(output_dir, "LogisticRegression", model_params,trainer.train_loss, trainer.train_accuracy, trainer.val_loss, trainer.val_accuracy)

    elif model_type == ModelType.DECISION_TREE:
        print("Decision Tree modeli eğitiliyor...")
        dt_trainer = DecisionTreeTrainer(model_params["max_depth"])
        dt_trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        print("Decision Tree modeli değerlendiriliyor...")
        y_pred_dt = dt_trainer.predict(X_test)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        print(f"Decision Tree Test Doğruluğu: {accuracy_dt * 100:.2f}%")
        
        save_results_to_excel(output_dir, "DecisionTree", model_params, dt_trainer.train_loss, dt_trainer.train_accuracy, dt_trainer.val_loss, dt_trainer.val_accuracy)

    elif model_type == ModelType.LSTM:
        print("LSTM modeli için veri hazırlanıyor...")
        
        time_steps = model_params["time_steps"]
        total_features = X_train.shape[1]

        # time_steps ile uyumlu num_features hesaplanıyor
        if total_features % time_steps != 0:
            raise ValueError(f"Özellik sayısı ({total_features}) time_steps ({time_steps}) ile uyumlu değil. time_steps değerini değiştirin.")
        
        num_features = total_features // time_steps

        # X_train ve X_test yeniden şekillendiriliyor
        X_train_lstm = X_train.reshape(-1, time_steps, num_features)
        X_test_lstm = X_test.reshape(-1, time_steps, num_features)

        print("LSTM modeli eğitiliyor...")
        lstm_trainer = LSTMTrainer(input_shape=(time_steps, num_features), lstm_units=64)
        lstm_trainer.train(
            X_train_lstm, y_train, X_val=X_test_lstm, y_val=y_test,
            epochs=model_params.get("epochs", 10),
            batch_size=model_params.get("batch_size", 32)
        )

        print("LSTM modeli değerlendiriliyor...")
        y_pred_lstm = lstm_trainer.predict(X_test_lstm)
        accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
        print(f"LSTM Test Doğruluğu: {accuracy_lstm * 100:.2f}%")

        save_results_to_excel(
            output_dir,
            "LSTM",
            model_params,
            lstm_trainer.history.history['loss'][-1],
            lstm_trainer.history.history['accuracy'][-1],
            lstm_trainer.history.history['val_loss'][-1],
            lstm_trainer.history.history['val_accuracy'][-1]
        )
        lstm_trainer.plot_metrics()
        
    elif model_type == ModelType.RANDOM_FOREST:
        print("Random Forest modeli eğitiliyor...")
        rf_trainer = RandomForestTrainer(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", None),
            random_state=model_params.get("random_state", 42)
        )
        rf_trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        print("Random Forest modeli değerlendiriliyor...")
        y_pred_rf = rf_trainer.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Test Doğruluğu: {accuracy_rf * 100:.2f}%")

        # Sonuçları kaydet
        save_results_to_excel(
            output_dir,
            "RandomForest",
            model_params,
            rf_trainer.train_loss,
            rf_trainer.train_accuracy,
            rf_trainer.val_loss,
            rf_trainer.val_accuracy
        )
        
    elif model_type == ModelType.SVM:
        print("SVM modeli eğitiliyor...")
        svm_trainer = SVMTrainer(
            kernel=model_params.get("kernel", "linear"),
            C=model_params.get("C", 1.0),
            random_state=model_params.get("random_state", 42)
        )
        svm_trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        print("SVM modeli değerlendiriliyor...")
        y_pred_svm = svm_trainer.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        print(f"SVM Test Doğruluğu: {accuracy_svm * 100:.2f}%")

        # Sonuçları kaydet
        save_results_to_excel(
            output_dir,
            "SVM",
            model_params,
            train_loss=None,  # SVM için `train_loss` hesaplanmaz
            train_accuracy=svm_trainer.train_accuracy,
            val_loss=None,  # SVM için `val_loss` hesaplanmaz
            val_accuracy=svm_trainer.val_accuracy
        )
        
    elif model_type == ModelType.ANN:
        print("Yapay Sinir Ağları (ANN) modeli eğitiliyor...")
        ann_trainer = ANNTrainer(
            input_dim=X_train.shape[1],
            hidden_layers=model_params.get("hidden_layers", [64, 32]),
            dropout_rate=model_params.get("dropout_rate", 0.2),
            learning_rate=model_params.get("learning_rate", 0.001)
        )
        ann_trainer.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=model_params.get("epochs", 10),
            batch_size=model_params.get("batch_size", 32)
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
            train_loss=ann_trainer.history.history['loss'][-1],
            train_accuracy=ann_trainer.history.history['accuracy'][-1],
            val_loss=ann_trainer.history.history['val_loss'][-1] if 'val_loss' in ann_trainer.history.history else None,
            val_accuracy=ann_trainer.history.history['val_accuracy'][-1] if 'val_accuracy' in ann_trainer.history.history else None
        )

    else:
        print("Geçersiz model türü seçildi!")

def main(file_path, selected_models):
    
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu.")

    # Veri yükleme
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)
    channels = [f"channel{i}" for i in range(1, 9)]

    # Veri temizleme
    print("Veri temizleme işlemi yapılıyor...")
    cleaner = DatasetCleaner()
    data = cleaner.drop_columns(data, columns=["label"])

    # Filtreleme işlemi
    print("Tüm kanallar için band geçiren filtre uygulanıyor...")
    filter_processor = DatasetFilter(data, channels, sampling_rate=1000)
    filter_processor.filter_all_channels(filter_type="band", cutoff=(1, 499), order=4)
    filtered_data = filter_processor.get_filtered_data()

    # Grafikler
    print("\nTüm kanallar için frekans spektrumları çiziliyor...")
    frequency_plot_path = os.path.join(output_dir, "frequency_spectra.png")
    filter_processor.plot_frequency_spectrum(
        signals=[data[channel] for channel in channels],
        filtered_signals=[filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Frekans Spektrumu" for channel in channels],
        output_path=frequency_plot_path
    )

    print("\nTüm kanallar için zaman domeni sinyalleri çiziliyor...")
    time_plot_path = os.path.join(output_dir, "time_domain_signals.png")
    filter_processor.plot_signals(
        signals=[data[channel] for channel in channels],
        filtered_signals=[filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Zaman Domeni Sinyalleri" for channel in channels],
        output_path=time_plot_path,
        start=0,
        end=1000
    )

    # Özellik çıkarma
    print("Özellikler çıkarılıyor...")
    features, labels = DatasetFeatureExtractor.extract_features(filtered_data, channels,window_size=200)

    # Veri dengeleme
    print("Veri SMOTE ile dengeleniyor...")
    balancer = DatasetBalancer()
    features, labels = balancer.balance(features, labels)

    # Veri ölçekleme
    print("Veri ölçekleniyor...")
    scaler = DatasetScaler()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Tüm Modelleri Eğitim Döngüsü
    for model_type, model_params in selected_models:
        run_model(model_type, model_params, X_train, y_train, X_test, y_test, output_dir="./output")
    
    # Sonuçların kaydedilmesi
    print(f"Model eğitimi ve değerlendirme başarıyla tamamlandı. Sonuçlar '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    
    dataset_path = "dataset/EMG-data.csv" 
    
    # Model parametreleri ve hangi modelin çalıştırılacağını belirleme
    selected_models = [
        (ModelType.LOGISTIC_REGRESSION, {"max_iter": 250}), # Logistic Regression
        (ModelType.DECISION_TREE, {"max_depth": 30}), # Decision Tree
        (ModelType.RANDOM_FOREST, { "n_estimators": 150,  "max_depth": 20, "random_state": 42}), # Random Forest
        (ModelType.ANN, {"hidden_layers": [32], "dropout_rate": 0.3, "learning_rate": 0.01, "epochs": 20, "batch_size": 64 }), # ANN
        (ModelType.LSTM, { "time_steps":8 , "lstm_units": 64,"epochs": 10, "batch_size": 90 }), # LSTM
        (ModelType.SVM, {"kernel": "linear", "C": 1.0,"random_state": 42}) # SVM 
    ]
    
    baslangic = time.time()
    main(dataset_path, selected_models=selected_models)
    bitis = time.time()
    
    # print("SVM Model :", bitis-baslangic)