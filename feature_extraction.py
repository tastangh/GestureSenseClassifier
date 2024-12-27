import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_features(segment, fs=200):
    """
    Verilen segmentten özellik çıkarımı yapar.
    :param segment: Tek bir zaman serisi segmenti (numpy array)
    :param fs: Örnekleme frekansı (Hz)
    :return: Özellik sözlüğü
    """
    features = {}
    
    # Zaman Domeni Özellikleri
    features["MAV"] = np.mean(np.abs(segment))  # Ortalama mutlak değer
    features["RMS"] = np.sqrt(np.mean(segment ** 2))  # Kök ortalama kare
    features["WL"] = np.sum(np.abs(np.diff(segment)))  # Dalga formu uzunluğu
    features["ZC"] = np.sum(np.diff(np.sign(segment)) != 0)  # Sıfır geçiş sayısı
    features["Skewness"] = skew(segment)  # Çarpıklık
    features["Kurtosis"] = kurtosis(segment)  # Basıklık
    features["Variance"] = np.var(segment)  # Varyans
    features["IEMG"] = np.sum(np.abs(segment))  # Entegre EMG
    
    # Frekans Domeni Özellikleri
    freqs, psd = welch(segment, fs=fs, nperseg=len(segment))  # Güç spektrumu yoğunluğu
    features["Mean_Frequency"] = np.sum(freqs * psd) / np.sum(psd)  # Ortalama frekans
    features["Median_Frequency"] = freqs[np.cumsum(psd) >= np.sum(psd) / 2][0]  # Medyan frekans
    features["Spectral_Entropy"] = -np.sum(psd * np.log(psd + 1e-10))  # Spektral entropi
    features["Spectral_Energy"] = np.sum(psd ** 2)  # Spektral enerji

    return features

def feature_extraction(data, fs=200):
    """
    Verilen veri setinden özellik çıkarımı yapar.
    :param data: Segmentlenmiş veri seti (pandas DataFrame)
    :param fs: Örnekleme frekansı (Hz)
    :return: Özellik DataFrame'i
    """
    feature_list = []
    labels = data["Gesture_Class"]
    for i, row in data.iterrows():
        segment = row[:-1].values  # Gesture_Class hariç sütunlar
        features = extract_features(segment, fs=fs)
        features["Gesture_Class"] = labels.iloc[i]  # Etiket eklenir
        feature_list.append(features)
    return pd.DataFrame(feature_list)

if __name__ == "__main__":
    # Girdi ve çıktı dosyaları
    input_file = "dataset/emg_data.csv"
    features_file = "features_emg_data.csv"

    # Veri yükleme ve özellik çıkarımı
    segmented_data = pd.read_csv(input_file)
    features = feature_extraction(segmented_data)

    # Özellikleri kaydetme
    features.to_csv(features_file, index=False)
    print(f"Özellikler çıkarıldı ve kaydedildi: {features_file}")
