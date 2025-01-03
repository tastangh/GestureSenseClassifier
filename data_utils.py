# data_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataset_cleaner import DatasetCleaner
from dataset_filter import DatasetFilter
from dataset_feature_extractor import DatasetFeatureExtractor
from dataset_balancer import DatasetBalancer
from dataset_scaler import DatasetScaler

def load_and_preprocess_data(file_path, filter_params, feature_params, channels, scaler_type="StandardScaler", balancer_type="SMOTE", data_pipeline_config=None):
    """
    Loads, preprocesses data and splits into train, validation, and test sets.
    :param file_path: Path to the data file.
    :param filter_params: Filter parameters.
    :param feature_params: Feature parameters.
    :param channels: List of channels.
    :param scaler_type: Type of scaler.
    :param balancer_type: Type of balancer.
    :param data_pipeline_config: Data processing pipeline configuration
    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """
    if data_pipeline_config is None:
          data_pipeline_config = {
              "clean_data": True,
              "filter_data": True,
              "extract_features": True,
              "balance_data": True,
              "scale_data": True
          }
    print("Veri yükleniyor...")
    data = pd.read_csv(file_path)

    # Veri temizleme
    if data_pipeline_config["clean_data"]:
         print("Veri temizleme işlemi yapılıyor...")
         cleaner = DatasetCleaner()
         data = cleaner.drop_columns(data, columns=["label"])

    # Split the data into train, validation and test sets
    print("Veri train, validation ve test olarak ayrılıyor...")
    
    # Split the data into train and test first
    train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['class'], random_state=42)
    
    # Then, split the remaining test_data into test and validation
    val_data, test_data = train_test_split(test_data, test_size=0.5, stratify=test_data['class'], random_state=42) 

    # Filtreleme işlemi
    if data_pipeline_config["filter_data"]:
        print("Tüm kanallar için filtreleme işlemi yapılıyor...")
        train_filtered_data = _filter_data(train_data, channels, filter_params)
        val_filtered_data = _filter_data(val_data, channels, filter_params)
        test_filtered_data = _filter_data(test_data, channels, filter_params)
    else :
        train_filtered_data = train_data
        val_filtered_data = val_data
        test_filtered_data = test_data

     # Veri setine 'class' sütununu ekleyelim.
    train_filtered_data['class'] = train_data['class']
    val_filtered_data['class'] = val_data['class']
    test_filtered_data['class'] = test_data['class']
    
    # Özellik çıkarma
    if data_pipeline_config["extract_features"]:
        print("Özellikler çıkarılıyor...")
        X_train, y_train = DatasetFeatureExtractor.extract_features(train_filtered_data, channels, window_size=feature_params["window_size"])
        X_val, y_val = DatasetFeatureExtractor.extract_features(val_filtered_data, channels, window_size=feature_params["window_size"])
        X_test, y_test = DatasetFeatureExtractor.extract_features(test_filtered_data, channels, window_size=feature_params["window_size"])
        
        # Handle NaN or infinite values
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    else :
        X_train = train_filtered_data[channels].values
        y_train = train_filtered_data['class'].values
        X_val = val_filtered_data[channels].values
        y_val = val_filtered_data['class'].values
        X_test = test_filtered_data[channels].values
        y_test = test_filtered_data['class'].values
    # Veri dengeleme
    if data_pipeline_config["balance_data"]:
       print("Veri SMOTE ile dengeleniyor...")
       balancer = DatasetBalancer()
       X_train, y_train = balancer.balance(X_train, y_train)
       X_val, y_val = balancer.balance(X_val, y_val)

    # Veri ölçekleme
    if data_pipeline_config["scale_data"]:
        print("Veri ölçekleniyor...")
        scaler = DatasetScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def _filter_data(data, channels, filter_params):
    """
    Applies filters to the dataset and returns filtered data.
    :param data: Data to filter.
    :param channels: Channels to filter.
    :param filter_params: Filtering parameters.
    :return: Filtered data
    """
    filter_processor = DatasetFilter(data, channels, sampling_rate=1000)
    filter_processor.filter_all_channels(
        filter_type=filter_params["filter_type"],
        cutoff=filter_params["cutoff"],
        order=filter_params["order"],
        apply_notch=filter_params["apply_notch"],
        notch_freq=filter_params["notch_freq"],
    )
    return filter_processor.get_filtered_data()