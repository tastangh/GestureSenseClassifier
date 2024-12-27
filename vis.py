import pandas as pd
import numpy as np

def compute_fft_features(data, fs=200, class_column="Gesture_Class"):
    """
    Zaman serisi verisinden frekans-genlik temelli özellikler çıkarır.
    :param data: Zaman serisi veri seti (Pandas DataFrame)
    :param fs: Örnekleme frekansı (Hz)
    :param class_column: Sınıf etiketi sütununun adı
    :return: Frekans özelliklerini içeren yeni bir DataFrame
    """
    sensor_columns = [col for col in data.columns if col != class_column]
    gesture_classes = data[class_column]
    
    # Yeni özellikler için DataFrame oluştur
    freq_features = []
    
    for idx, row in data.iterrows():
        # Sensör verisini yeniden şekillendir (8 sensör, 8 zaman noktası)
        sensors = np.array(row[sensor_columns]).reshape(8, 8)
        row_features = []
        
        for sensor in sensors:
            # FFT uygula
            fft_vals = np.abs(np.fft.rfft(sensor))
            fft_freqs = np.fft.rfftfreq(len(sensor), d=1/fs)
            
            # Özellikler çıkar
            row_features.append(np.mean(fft_vals))  # Ortalama genlik
            row_features.append(np.max(fft_vals))   # Maksimum genlik
            row_features.append(np.sum(fft_vals))  # Toplam enerji
            row_features.append(fft_freqs[np.argmax(fft_vals)])  # Tepe frekansı

        # Özellikleri birleştir
        freq_features.append(row_features)
    
    # Sınıf etiketini ekle
    freq_features_df = pd.DataFrame(freq_features, columns=[
        f"Sensor_{i+1}_{stat}" for i in range(8) for stat in ["Mean", "Max", "Sum", "PeakFreq"]
    ])
    freq_features_df[class_column] = gesture_classes.values

    return freq_features_df

# Kullanım Örneği
if __name__ == "__main__":
    input_file = "dataset/emg_data.csv"  # Veri setinin doğru yolunu kontrol edin
    output_file = "frequency_based_dataset.csv"
    
    # Veriyi yükle
    try:
        data = pd.read_csv(input_file)
        print(f"Veri Yüklendi: {data.shape}")
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        exit()
    
    # Frekans özelliklerini çıkar
    print("Frekans tabanlı özellikler çıkarılıyor...")
    freq_features_df = compute_fft_features(data, fs=200, class_column="Gesture_Class")
    
    # Yeni veri setini kaydet
    freq_features_df.to_csv(output_file, index=False)
    print(f"Frekans tabanlı veri seti kaydedildi: {output_file}")
