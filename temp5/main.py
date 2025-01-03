import os
import time
import datetime
import numpy as np
from enum import Enum

from data_utils import load_and_preprocess_data
from param_utils import create_parameter_combinations
from save_results import save_results
from base_trainer import BaseTrainer  # Import BaseTrainer
from sklearn.metrics import accuracy_score
# Models
from decisiontree_trainer import DecisionTreeTrainer
from lstm_trainer import LSTMTrainer
from log_reg_trainer import LogRegTrainer
from randomforest_trainer import RandomForestTrainer
from svm_trainer import SVMTrainer
from ann_trainer import ANNTrainer
import tensorflow as tf

# TensorFlow GPU memory limit configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use 10GB of GPU RAM
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

class ModelType(Enum):
    LOGISTIC_REGRESSION = "LogisticRegression"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    LSTM = "LSTM"
    SVM = "SVM"
    # Artificial Neural Networks - Yapay Sinir Ağları
    ANN = "ANN"

def run_model(model_type, model_params, X_train, y_train, X_test, y_test, output_dir, filter_params, feature_params, scaler_type, balancer_type, data_pipeline_config):
    """
    Trains, evaluates, and saves results for a given model type.
    :param model_type: Type of the model to train.
    :param model_params: Parameters for the model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param output_dir: Directory to save results.
    :param filter_params: Filtering parameters.
    :param feature_params: Feature extraction parameters.
    :param scaler_type: Type of scaler.
    :param balancer_type: Type of balancer.
    :param data_pipeline_config: Data processing pipeline configuration
    """
    if model_type == ModelType.LOGISTIC_REGRESSION:
        trainer = LogRegTrainer(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)), output_dir=output_dir)
    elif model_type == ModelType.DECISION_TREE:
        trainer = DecisionTreeTrainer(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)), max_depth=model_params.get("max_depth"), output_dir=output_dir)
    elif model_type == ModelType.LSTM:
        trainer = LSTMTrainer(input_shape=(X_train.shape[1] // model_params.get("time_steps", 1), model_params.get("time_steps", 1)),
                              num_classes=len(np.unique(y_train)),
                              lstm_units=model_params.get("lstm_units", 64),
                              output_dir=output_dir)
        # LSTM için veriyi yeniden şekillendir
        time_steps = model_params["time_steps"]
        total_features = X_train.shape[1]

        # time_steps ile uyumlu num_features hesaplanıyor
        if total_features % time_steps != 0:
            raise ValueError(
                f"Özellik sayısı ({total_features}) time_steps ({time_steps}) ile uyumlu değil. time_steps değerini değiştirin.")

        num_features = total_features // time_steps
        X_train = X_train.reshape(-1, time_steps, num_features)
        X_test = X_test.reshape(-1, time_steps, num_features)
    elif model_type == ModelType.RANDOM_FOREST:
        trainer = RandomForestTrainer(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)),
                                       n_estimators=model_params.get("n_estimators", 100),
                                        max_depth=model_params.get("max_depth"), output_dir=output_dir)
    elif model_type == ModelType.SVM:
        trainer = SVMTrainer(input_dim=X_train.shape[1], num_classes = len(np.unique(y_train)),
            C=model_params.get("C", 1.0), output_dir=output_dir
        )
    elif model_type == ModelType.ANN:
          trainer = ANNTrainer(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            hidden_layers=model_params.get("hidden_layers", [64, 32]),
            dropout_rate=model_params.get("dropout_rate", 0.2),
            learning_rate=model_params.get("learning_rate", 0.001),
            output_dir=output_dir
        )

    else:
        print("Geçersiz model türü seçildi!")
        return

    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test,
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
    
    y_pred = trainer.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_type.value} Test Doğruluğu: {test_accuracy * 100:.2f}%")

    save_results(
        output_dir,
        model_type.value,
        model_params,
        trainer.history['loss'][-1],
        trainer.history['accuracy'][-1],
        trainer.history['val_loss'][-1] if 'val_loss' in trainer.history else None,
        trainer.history['val_accuracy'][-1] if 'val_accuracy' in trainer.history else None,
        test_accuracy,
        filter_params,
        feature_params,
        scaler_type,
        balancer_type,
        format="excel"
    )
    trainer.plot_metrics(model_type.value, model_params)

def main(file_path, selected_models, filter_params, feature_params, scaler_type="StandardScaler", balancer_type="SMOTE", data_pipeline_config=None):
    """
    Main function to orchestrate the loading, preprocessing, and training of models.
    :param file_path: Path to the data file.
    :param selected_models: List of tuples with model type and parameters.
    :param filter_params: Filter parameters.
    :param feature_params: Feature extraction parameters.
    :param scaler_type: Type of scaler.
    :param balancer_type: Type of balancer.
    :param data_pipeline_config: Data processing pipeline configuration
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", timestamp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        file_path, filter_params, feature_params,
        channels=[f"channel{i}" for i in range(1, 9)],
        scaler_type=scaler_type,
        balancer_type=balancer_type,
        data_pipeline_config=data_pipeline_config
    )
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    all_model_params = create_parameter_combinations(selected_models)

    for model_type, model_params in all_model_params:
        run_model(model_type, model_params, X_train, y_train, X_test, y_test,
                  output_dir, filter_params, feature_params, scaler_type, balancer_type, data_pipeline_config)

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

    feature_params = {"window_size": 300}

      # Data pipeline configuration
    data_pipeline_config = {
        "clean_data": True,
        "filter_data": True,
        "extract_features": True,
        "balance_data": True,
        "scale_data": True
    }

    # Model parametreleri ve hangi modelin çalıştırılacağını belirleme
    selected_models = [
        (ModelType.LOGISTIC_REGRESSION, {
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 32,
            "optimizer_type": "adam",
            "early_stopping": True,
            "patience": 10,
            "learning_rate_scheduling": True,
            "factor": 0.1,
            "min_lr": 1e-6
        }),
        (ModelType.DECISION_TREE, {"max_depth": 30}), # Decision Tree
        (ModelType.RANDOM_FOREST, { "n_estimators": 150,  "max_depth": 20, "random_state": 42}), # Random Forest
         (ModelType.ANN, {"hidden_layers": [32], "dropout_rate": 0.3, "learning_rate": 0.01, "epochs": 20, "batch_size": 64 ,
                          "early_stopping": True,
                          "patience": 10,
                          "learning_rate_scheduling": True,
                          "factor": 0.1,
                          "min_lr": 1e-6}), # ANN
        (ModelType.LSTM, { "time_steps":8 , "lstm_units": 64,"epochs": 10, "batch_size": 90 ,
                           "learning_rate": 0.001,
                           "early_stopping": True,
                           "patience": 10,
                           "learning_rate_scheduling": True,
                           "factor": 0.1,
                           "min_lr": 1e-6}), # LSTM
       (ModelType.SVM, {"kernel": "linear", "C": 1.0,"random_state": 42}) # SVM
    ]

    baslangic = time.time()
    main(dataset_path, selected_models, filter_params, feature_params, data_pipeline_config=data_pipeline_config)
    bitis = time.time()

    print("Toplam geçen süre :", bitis - baslangic)