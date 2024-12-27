import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=4):
    """
    Düşük geçiren filtre (Lowpass) oluşturur.
    :param cutoff: Kesme frekansı (Hz).
    :param fs: Örnekleme frekansı (Hz).
    :param order: Filtrenin derecesi.
    """
    nyquist = 0.5 * fs  # Nyquist frekansı
    if cutoff >= nyquist:
        raise ValueError(f"Kesme frekansı ({cutoff} Hz), Nyquist frekansından ({nyquist} Hz) küçük olmalıdır.")
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=4):
    """
    Band-stop filtre oluşturur (ör. 50 Hz gürültüsü için).
    :param lowcut: Alt kesme frekansı (Hz).
    :param highcut: Üst kesme frekansı (Hz).
    :param fs: Örnekleme frekansı (Hz).
    :param order: Filtrenin derecesi.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    return b, a

def apply_filters(data, fs=200):
    """
    Veriye düşük geçiren ve band-stop filtre uygular.
    :param data: Zaman serisi veri seti (Pandas DataFrame).
    :param fs: Örnekleme frekansı (Hz).
    """
    sensor_columns = [col for col in data.columns if col != "Gesture_Class"]
    
    # Düşük geçiren filtre
    print("Düşük geçiren filtre uygulanıyor (100 Hz)...")
    b_low, a_low = butter_lowpass(99, fs)
    filtered_data = data[sensor_columns].apply(lambda x: filtfilt(b_low, a_low, x), axis=0)
    
    # Band-stop filtre
    print("Band-stop filtre uygulanıyor (48-52 Hz)...")
    b_stop, a_stop = butter_bandstop(48, 52, fs)
    filtered_data = filtered_data.apply(lambda x: filtfilt(b_stop, a_stop, x), axis=0)
    
    # Sınıf etiketini ekle
    filtered_data["Gesture_Class"] = data["Gesture_Class"]
    return filtered_data

if __name__ == "__main__":
    # Veriyi yükle
    input_file = "dataset/emg_data.csv"
    try:
        data = pd.read_csv(input_file)
        print(f"Veri başarıyla yüklendi: {data.shape}")
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        exit()

    # Filtreleme işlemi
    print("Filtreleme işlemi başlatılıyor...")
    filtered_data = apply_filters(data, fs=200)

    # Filtrelenmiş veriyi kaydet
    output_file = "filtered_emg_data.csv"
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtrelenmiş veri kaydedildi: {output_file}")
