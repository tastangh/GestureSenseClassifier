# dataset_feature_extractor.py
import numpy as np

class DatasetFeatureExtractor:
    @staticmethod
    def extract_features(data, channels, window_size=200):
        features = []
        labels = []
        for gesture in data["class"].unique():
            subset = data[data["class"] == gesture]
            for i in range(0, len(subset) - window_size, window_size):
                window = subset.iloc[i : i + window_size][channels].values
                features.append(DatasetFeatureExtractor.advanced_feature_extraction(window))
                labels.append(gesture)
        return np.array(features), np.array(labels)

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