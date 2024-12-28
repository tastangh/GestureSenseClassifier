import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, lfilter
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Zaman Serisi Filtreleme (Düşük Geçiren Filtre)
def low_pass_filter(data, cutoff=0.1, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)

# 2. Gelişmiş Özellik Çıkarma Fonksiyonu
def advanced_feature_extraction(window):
    feature_vector = {
        "mean": window.mean(axis=0),
        "std": window.std(axis=0),
        "min": window.min(axis=0),
        "max": window.max(axis=0),
        "energy": np.sum(np.square(window), axis=0),
        "zero_crossing_rate": np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0),
        "dominant_frequency": np.argmax(np.abs(np.fft.fft(window, axis=0)), axis=0),
    }
    return np.concatenate(list(feature_vector.values()))

def extract_features(data, window_size=200):
    features = []
    labels = []
    for label in data["label"].unique():
        for gesture in data[data["label"] == label]["class"].unique():
            subset = data[(data["label"] == label) & (data["class"] == gesture)]
            for i in range(0, len(subset) - window_size, window_size):
                window = subset.iloc[i : i + window_size, 1:9].values  # Sadece EMG kanalları
                features.append(advanced_feature_extraction(window))
                labels.append(gesture)
    return np.array(features), np.array(labels)

# 3. Veri Yükleme ve Hazırlık
data = pd.read_csv("dataset/EMG-data.csv")  # Dataset yolunu güncelleyin
for col in [f"channel{i}" for i in range(1, 9)]:
    data[col] = low_pass_filter(data[col].values)  # Düşük geçiren filtre

# Özellik çıkarımı
features, labels = extract_features(data)

# Veri dengesizliğini dengelemek için SMOTE
smote = SMOTE(random_state=42)
features, labels = smote.fit_resample(features, labels)

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Confusion Matrix Görselleştirme Fonksiyonu
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Gerçek Sınıf")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.show()

# 4. Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
logistic_model.fit(X_train, y_train)
y_pred_lr = logistic_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

# 5. SVM
svm_model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
plot_confusion_matrix(y_test, y_pred_svm, "SVM")

# 6. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# 7. LSTM
# LSTM için veriyi 3D şekline dönüştürme
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_train_lstm = to_categorical(y_train)
y_test_lstm = to_categorical(y_test)

# LSTM modeli tanımlama
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(y_train_lstm.shape[1], activation="softmax")
])
lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

# LSTM tahminleri ve doğruluğu
y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
accuracy_lstm = accuracy_score(np.argmax(y_test_lstm, axis=1), y_pred_lstm)
plot_confusion_matrix(np.argmax(y_test_lstm, axis=1), y_pred_lstm, "LSTM")

# 8. Sonuçların Yazdırılması
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print(f"LSTM Accuracy: {accuracy_lstm * 100:.2f}%")
