import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, lfilter
import seaborn as sns
import matplotlib.pyplot as plt

# # 1. Zaman Serisi Filtreleme (Düşük Geçiren Filtre)
# def low_pass_filter(data, cutoff=50, fs=1000, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype="low", analog=False)
#     return lfilter(b, a, data)

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
        "rms": np.sqrt(np.mean(np.square(window), axis=0)),
        "waveform_length": np.sum(np.abs(np.diff(window, axis=0)), axis=0),
    }
    return np.concatenate(list(feature_vector.values()))

# 3. Özellik Çıkarma Fonksiyonu
def extract_features(data, window_size=200):
    features = []
    labels = []
    for gesture in data["class"].unique():
        subset = data[data["class"] == gesture]
        for i in range(0, len(subset) - window_size, window_size):
            window = subset.iloc[i : i + window_size, 1:9].values  # Sadece EMG kanalları
            features.append(advanced_feature_extraction(window))
            labels.append(gesture)
    return np.array(features), np.array(labels)

# 4. Veri Yükleme ve Hazırlık
data = pd.read_csv("dataset/EMG-data.csv")  # Dataset yolunu güncelleyin

# 'label' sütununu veri setinden çıkarın
data = data.drop(columns=["label"])

# # EMG sinyallerine düşük geçiren filtre uygulama
# for col in [f"channel{i}" for i in range(1, 9)]:
#     data[col] = low_pass_filter(data[col].values)

# Özellik çıkarımı
features, labels = extract_features(data)

# 5. Veri dengesizliğini dengelemek için SMOTE
smote = SMOTE(random_state=42)
features, labels = smote.fit_resample(features, labels)

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

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

# 6. Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
logistic_model.fit(X_train, y_train)
y_pred_lr = logistic_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

# 7. Sonuçların Yazdırılması
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
