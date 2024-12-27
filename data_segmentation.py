# data_segmentation.py
import pandas as pd

def segment_data(data, window_size=60, overlap=10):
    sensor_columns = [col for col in data.columns if col != "Gesture_Class"]
    gesture_class = data["Gesture_Class"]
    segments = []
    labels = []

    for i in range(0, len(data) - window_size, window_size - overlap):
        segment = data[sensor_columns].iloc[i:i + window_size].values.flatten()
        segments.append(segment)
        labels.append(gesture_class.iloc[i])

    segment_df = pd.DataFrame(segments)
    segment_df["Gesture_Class"] = labels
    return segment_df

if __name__ == "__main__":
    input_file = "filtered_emg_data.csv"
    filtered_data = pd.read_csv(input_file)
    segmented_data = segment_data(filtered_data, window_size=60, overlap=10)
    segmented_file = "segmented_emg_data.csv"
    segmented_data.to_csv(segmented_file, index=False)
    print(f"Segmentlere ayrılmış veri kaydedildi: {segmented_file}")
