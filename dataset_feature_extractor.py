import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class DatasetFeatureExtractor:
    @staticmethod
    def extract_features(data, channels, window_size=200, feature_set="advanced", n_components=None, k_best=None):
        features = []
        labels = []
        for gesture in data["class"].unique():
            subset = data[data["class"] == gesture]
            for i in range(0, len(subset) - window_size, window_size):
                window = subset.iloc[i : i + window_size][channels].values
                if feature_set == "advanced":
                    extracted_features = DatasetFeatureExtractor.advanced_feature_extraction(window)
                elif feature_set == "time":
                     extracted_features = DatasetFeatureExtractor.time_domain_feature_extraction(window)
                elif feature_set == "frequency":
                     extracted_features = DatasetFeatureExtractor.frequency_domain_feature_extraction(window)
                elif feature_set == "basic":
                    extracted_features = DatasetFeatureExtractor.basic_feature_extraction(window)
                else:
                    raise ValueError("Geçersiz özellik seti seçildi! 'advanced', 'basic', 'frequency', 'time' kullanın.")
                    
                features.append(extracted_features)
                labels.append(gesture)

        features = np.array(features)

        if n_components and feature_set !="frequency" :
             features = DatasetFeatureExtractor.apply_pca(features, n_components)
        elif k_best and feature_set != "frequency":
             features = DatasetFeatureExtractor.apply_select_k_best(features, labels, k_best)

        return np.array(features), np.array(labels)


    @staticmethod
    def basic_feature_extraction(window):
          feature_vector = {
           "mean": window.mean(axis=0),
           "std": window.std(axis=0),
           "min": window.min(axis=0),
           "max": window.max(axis=0),
            "rms": np.sqrt(np.mean(np.square(window), axis=0)),
           "waveform_length": np.sum(np.abs(np.diff(window, axis=0)), axis=0),  # WL
         }
          return np.concatenate(list(feature_vector.values()))

    @staticmethod
    def advanced_feature_extraction(window):
        # Zaman ve frekans özellikleri
        feature_vector = {
            "mean": window.mean(axis=0),
            "std": window.std(axis=0),
            "min": window.min(axis=0),
            "max": window.max(axis=0),
            "energy": np.sum(np.square(window), axis=0),  # Enerji
            "zero_crossing_rate": np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0),  # ZC
            "dominant_frequency": np.argmax(np.abs(np.fft.fft(window, axis=0)), axis=0),  # Dominant frekans
            "rms": np.sqrt(np.mean(np.square(window), axis=0)),  # RMS
            "waveform_length": np.sum(np.abs(np.diff(window, axis=0)), axis=0),  # WL
            "mean_absolute_value": np.mean(np.abs(window), axis=0),  # MAV
            "variance": np.var(window, axis=0),  # VAR
            "slope_sign_changes": np.sum(np.diff(np.sign(np.diff(window, axis=0)), axis=0) != 0, axis=0),  # SSC
            "willison_amplitude": np.sum(np.abs(np.diff(window, axis=0)) > 0.01, axis=0),  # WAMP
            "myopulse_percentage_rate": np.sum(np.abs(window) > 0.01, axis=0) / window.shape[0],  # MYOP
            "simple_square_integral": np.sum(window**2, axis=0),  # SSI
            "log_detector": np.exp(np.mean(np.log(np.abs(window) + 1e-10), axis=0)),  # LD
            "difference_absolute_standard_deviation_value": np.mean(np.abs(np.diff(window, axis=0)), axis=0),  # DASDV
            "maximum_fractal_length": np.sum(np.abs(np.diff(window, axis=0))**1.5, axis=0)  # MFL
        }
        # Tüm özellikleri birleştir
        return np.concatenate(list(feature_vector.values()))

    @staticmethod
    def frequency_domain_feature_extraction(window, sampling_rate=1000):
        feature_vector = {
        "band_power": DatasetFeatureExtractor.extract_band_powers(window, sampling_rate, band_ranges=[(20, 75),(75, 150),(150, 300), (300, 450)]),
         "dominant_frequency": np.argmax(np.abs(np.fft.fft(window, axis=0)), axis=0)
         }
        return np.concatenate(list(feature_vector.values()))

    @staticmethod
    def time_domain_feature_extraction(window):
           feature_vector = {
           "mean": window.mean(axis=0),
           "std": window.std(axis=0),
           "min": window.min(axis=0),
           "max": window.max(axis=0),
           "rms": np.sqrt(np.mean(np.square(window), axis=0)),
           "waveform_length": np.sum(np.abs(np.diff(window, axis=0)), axis=0),
           "mean_absolute_value": np.mean(np.abs(window), axis=0),
           "variance": np.var(window, axis=0),
           "willison_amplitude": np.sum(np.abs(np.diff(window, axis=0)) > 0.01, axis=0),
           "myopulse_percentage_rate": np.sum(np.abs(window) > 0.01, axis=0) / window.shape[0],
          }
           return np.concatenate(list(feature_vector.values()))

    @staticmethod
    def extract_band_powers(window, sampling_rate, band_ranges):
        """Belirli frekans bantlarındaki güçleri hesaplar."""
        f, psd = welch(window, sampling_rate, nperseg=len(window))
        band_powers = []
        for low, high in band_ranges:
            band_power = np.sum(psd[(f >= low) & (f <= high)], axis = 0)
            band_powers.append(band_power)
        return np.array(band_powers)

    @staticmethod
    def apply_pca(features, n_components):
        """Veriye PCA uygular."""
        print(f"PCA uygulanıyor, bileşen sayısı: {n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(features)

    @staticmethod
    def apply_select_k_best(features, labels, k_best):
        """Veriye SelectKBest uygular."""
        print(f"SelectKBest uygulanıyor, en iyi özellik sayısı: {k_best}")
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selected_features = selector.fit_transform(features, labels)
        return selected_features