import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

def apply_frequency_filter(signal, fs=200, lowcut=10, highcut=99):
    """
    Frekans düzleminde filtre uygular.
    :param signal: Zaman serisi sinyali.
    :param fs: Örnekleme frekansı (Hz).
    :param lowcut: Alt kesme frekansı (Hz).
    :param highcut: Üst kesme frekansı (Hz).
    :return: Filtrelenmiş sinyal.
    """
    # FFT uygula
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    fft_vals = fft(signal)

    # Frekansları filtrele
    filtered_fft_vals = np.where((freqs >= lowcut) & (freqs <= highcut), fft_vals, 0)

    # Ters FFT ile zaman düzlemine dön
    filtered_signal = np.real(ifft(filtered_fft_vals))
    return filtered_signal

def plot_signal_and_fft(signal, fs=200):
    """
    Zaman serisi ve frekans spektrumunu çizer.
    :param signal: Zaman serisi sinyali.
    :param fs: Örnekleme frekansı (Hz).
    """
    # FFT hesapla
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    fft_vals = np.abs(fft(signal))

    # Zaman serisini çiz
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title("Zaman Serisi")
    plt.xlabel("Zaman")
    plt.ylabel("Genlik")

    # Frekans spektrumunu çiz
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:N // 2], fft_vals[:N // 2])  # Pozitif frekanslar
    plt.title("Frekans Spektrumu")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Genlik")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Örnek veri (1 sensör)
    input_file = "dataset/emg_data.csv"
    data = pd.read_csv(input_file)

    # İlk sensörün ilk sütununu al
    signal = data.iloc[:, 0].values

    # Zaman serisi ve FFT grafiği
    print("Ham sinyal:")
    plot_signal_and_fft(signal)

    # Frekans düzleminde filtreleme
    print("Frekans düzleminde filtrelenmiş sinyal:")
    filtered_signal = apply_frequency_filter(signal, fs=200, lowcut=10, highcut=99)
    plot_signal_and_fft(filtered_signal)

    # Tüm sensörlere filtre uygulama
    print("Tüm sensörlere filtre uygulanıyor...")
    sensor_data = data.iloc[:, :-1]
    filtered_data = sensor_data.apply(lambda col: apply_frequency_filter(col.values, fs=200, lowcut=10, highcut=99), axis=0)

    # Sonuçları kaydet
    filtered_data["Gesture_Class"] = data["Gesture_Class"]
    output_file = "filtered_emg_data_frequency_domain.csv"
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtrelenmiş veri kaydedildi: {output_file}")
