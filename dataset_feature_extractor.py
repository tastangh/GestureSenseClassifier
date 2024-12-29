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
