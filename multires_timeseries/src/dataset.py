import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, Subset

def get_time_split(dataset, split_ratio=0.8):
    split_idx = int(len(dataset) * split_ratio)
    train_dataset = Subset(dataset, list(range(split_idx)))
    val_dataset = Subset(dataset, list(range(split_idx, len(dataset))))
    return train_dataset, val_dataset

def generate_ar_series(length, features, phi=0.9):
    # generating 4 traffic related synthetic features
    series = np.zeros((length, features))
    noise = np.random.randn(length, features)
    for t in range(1, length):
        series[t] = phi * series[t - 1] + noise[t]
    return series

def seasonal_component(length, period=24, amplitude=5):
    return amplitude * np.sin(np.arange(length) * 2 * np.pi / period)

def time_features(index):
    # generating 4 time features
    hour = index % 24
    day = (index // 24) % 7
    return [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 7),
        np.cos(2 * np.pi * day / 7)
    ]

# Dataset using AR series with seasonal component, overlapping windows, and time features
class TrafficDataset(Dataset):
    def __init__(self, num_days=90, hourly_len=168, fivemin_len=72, features=4, stride=12):
        num_samples = num_days * 24  # one per hour
        # 4 synthetic features being generated (e.g. vehicle count, occupancy rate) per time
        base_series = generate_ar_series(num_samples, features)
        seasonal = seasonal_component(num_samples).reshape(-1, 1)
        data_hourly = base_series + seasonal

        # Normalize
        mean = np.mean(data_hourly, axis=0)
        std = np.std(data_hourly, axis=0) + 1e-6
        data_hourly = (data_hourly - mean) / std

        self.hourly_data = []
        self.fivemin_data = []
        self.y_5min = []
        self.y_hourly = []

        for i in range(hourly_len, num_samples - fivemin_len, stride):
            hourly_seq = data_hourly[i-hourly_len:i]
            fivemin_seq = np.repeat(data_hourly[i:i+fivemin_len//12], 12, axis=0) + 0.01 * np.random.randn(fivemin_len, features)

            # Add time features
            hourly_tf = np.array([time_features(i - hourly_len + j) for j in range(hourly_len)])
            fivemin_tf = np.array([time_features(i + j // 12) for j in range(fivemin_len)])
            hourly_seq = np.concatenate([hourly_seq, hourly_tf], axis=1)
            fivemin_seq = np.concatenate([fivemin_seq, fivemin_tf], axis=1)

            y_5min = np.sum(fivemin_seq[:, 0:1], axis=0, keepdims=True).repeat(fivemin_len, axis=0)
            y_hourly = np.sum(hourly_seq[:, 0:1])

            self.hourly_data.append(hourly_seq.astype(np.float32))
            self.fivemin_data.append(fivemin_seq.astype(np.float32))
            self.y_5min.append(y_5min.astype(np.float32))
            self.y_hourly.append(np.array([y_hourly], dtype=np.float32))

    def __len__(self):
        return len(self.hourly_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.hourly_data[idx]),
            torch.tensor(self.fivemin_data[idx]),
            torch.tensor(self.y_5min[idx]),
            torch.tensor(self.y_hourly[idx]),
        )