import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import h2o
from h2o.estimators import H2OSupportVectorMachineEstimator
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# H2O.ai başlatma
h2o.init()

# Dosya yolları
RAW_DATA_PATH = "dataset/emg_data.csv"
FEATURES_DATA_PATH = "features_emg_data.csv"
RESULTS_DIR = "results/model_comparisons"

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_logistic_regression(X_train, X_val, y_train, y_val):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy, classification_report(y_val, y_pred)

def train_h2o_svm(X_train, X_val, y_train, y_val):
    """
    H2O.ai ile SVM modeli eğitimi.
    """
    # Veriyi H2O çerçevesine dönüştür
    train_data = pd.DataFrame(X_train)
    train_data["Label"] = y_train
    train_h2o = h2o.H2OFrame(train_data)

    val_data = pd.DataFrame(X_val)
    val_data["Label"] = y_val
    val_h2o = h2o.H2OFrame(val_data)

    # SVM modeli eğitimi
    svm_model = H2OSupportVectorMachineEstimator(gamma=0.1, C=1)
    svm_model.train(x=train_h2o.columns[:-1], y="Label", training_frame=train_h2o)

    # Model değerlendirme
    performance = svm_model.model_performance(val_h2o)
    accuracy = performance.accuracy()[0][1]
    logger.info(f"SVM Doğruluk: {accuracy}")
    return accuracy, str(performance)

def train_lstm(X_train, X_val, y_train, y_val, input_dim):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = X_train.reshape(X_train.shape[0], 1, input_dim)
    X_val = X_val.reshape(X_val.shape[0], 1, input_dim)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    model = Sequential([
        LSTM(64, input_shape=(1, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return accuracy, "LSTM Model değerlendirme başarıyla tamamlandı."

def load_data_and_split(data_path, label_column="Gesture_Class", test_size=0.2):
    data = pd.read_csv(data_path)
    X = data.drop(columns=[label_column]).values
    y = data[label_column].values
    return train_test_split(X, y, test_size=test_size, random_state=42)

def evaluate_models(data_path, dataset_name):
    print(f"\n--- {dataset_name} Veri Seti ile Model Eğitimi ve Değerlendirme ---")
    X_train, X_val, y_train, y_val = load_data_and_split(data_path)
    input_dim = X_train.shape[1]
    
    # Logistic Regression
    print("\nLogistic Regression:")
    lr_acc, lr_report = train_logistic_regression(X_train, X_val, y_train, y_val)
    print(f"Doğruluk: {lr_acc}\n{lr_report}")
    
    # H2O SVM
    print("\nH2O SVM:")
    svm_acc, svm_report = train_h2o_svm(X_train, X_val, y_train, y_val)
    print(f"Doğruluk: {svm_acc}\n{svm_report}")
    
    # LSTM
    print("\nLSTM:")
    lstm_acc, lstm_report = train_lstm(X_train, X_val, y_train, y_val, input_dim)
    print(f"Doğruluk: {lstm_acc}\n{lstm_report}")

    return {
        "Logistic Regression": lr_acc,
        "H2O SVM": svm_acc,
        "LSTM": lstm_acc
    }

def main():
    print("\n--- Ham Veri (emg_data.csv) Model Değerlendirmesi ---")
    raw_results = evaluate_models(RAW_DATA_PATH, "Ham Veri")
    
    print("\n--- Özellik Çıkarımı (features_emg_data.csv) Model Değerlendirmesi ---")
    features_results = evaluate_models(FEATURES_DATA_PATH, "Özellik Çıkarımı")
    
    print("\n--- SONUÇLAR KARŞILAŞTIRMASI ---")
    print("Ham Veri Sonuçları:", raw_results)
    print("Özellik Çıkarımı Sonuçları:", features_results)

if __name__ == "__main__":
    main()
