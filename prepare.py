import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_nan_sensor_data(df):
    df = df.interpolate(method='linear', limit_direction='both')
    return df.ffill().bfill()


# PAMAP2 칼럼: timestamp(0), activityID(1), heartRate(2), 
# IMU_hand(3-19), IMU_chest(20-36), IMU_ankle(37-53)
# 각 IMU 내부: temp(0), acc16g(1-3), acc6g(4-6), gyro(7-9), mag(10-12), orient(13-16)
PAMAP2_ACTIVITY_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
                       12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
PAMAP2_ACC_COLS = [4, 5, 6, 21, 22, 23, 38, 39, 40]
PAMAP2_GYRO_COLS = [10, 11, 12, 27, 28, 29, 44, 45, 46]


def load_pamap2(data_dir, window_size=128, step_size=64, test_subjects=(105, 106)):
    protocol_dir = os.path.join(data_dir, 'PAMAP2_Dataset', 'Protocol')
    use_cols = PAMAP2_ACC_COLS + PAMAP2_GYRO_COLS
    
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []

    for fname in sorted(os.listdir(protocol_dir)):
        if not fname.startswith('subject'):
            continue
        subj_id = int(fname.replace('subject', '').replace('.dat', ''))
        fpath = os.path.join(protocol_dir, fname)
        
        df = pd.read_csv(fpath, sep=r'\s+', header=None)
        
        df = df[df[1] != 0].reset_index(drop=True)
        
        sensor_df = df[use_cols].copy()
        sensor_df = clean_nan_sensor_data(sensor_df)
        labels = df[1].values
        sensor_data = sensor_df.values.astype(np.float32)
        
        mapped_labels = np.array([PAMAP2_ACTIVITY_MAP.get(l, -1) for l in labels])
        
        for start in range(0, len(sensor_data) - window_size + 1, step_size):
            end = start + window_size
            seg = sensor_data[start:end]
            seg_labels = mapped_labels[start:end]
            
            if (seg_labels == -1).any():
                continue
            
            label = np.bincount(seg_labels[seg_labels >= 0]).argmax()
            
            if subj_id in test_subjects:
                test_windows.append(seg)
                test_labels.append(label)
            else:
                train_windows.append(seg)
                train_labels.append(label)
    
    X_train = np.stack(train_windows)
    y_train = np.array(train_labels)
    X_test = np.stack(test_windows)
    y_test = np.array(test_labels)
    
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"[PAMAP2] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Features: {X_train.shape[2]}, Classes: {len(set(y_train))}")
    return X_train, y_train, X_test, y_test


HHAR_ACTIVITY_MAP = {'bike': 0, 'sit': 1, 'stand': 2, 
                     'walk': 3, 'stairsup': 4, 'stairsdown': 5}


def load_hhar(data_dir, window_size=128, step_size=64, test_users=('f', 'g', 'h', 'i')):
    acc_path = os.path.join(data_dir, 'Activity recognition exp', 'Phones_accelerometer.csv')
    gyro_path = os.path.join(data_dir, 'Activity recognition exp', 'Phones_gyroscope.csv')
    
    acc_df = pd.read_csv(acc_path)
    gyro_df = pd.read_csv(gyro_path)
    
    acc_df = acc_df.dropna(subset=['gt'])
    gyro_df = gyro_df.dropna(subset=['gt'])
    acc_df = acc_df[acc_df['gt'] != 'null'].reset_index(drop=True)
    gyro_df = gyro_df[gyro_df['gt'] != 'null'].reset_index(drop=True)
    
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []
    
    users = acc_df['User'].unique()
    devices = acc_df['Device'].unique()
    
    for user in sorted(users):
        for device in sorted(devices):
            a = acc_df[(acc_df['User'] == user) & (acc_df['Device'] == device)].copy()
            g = gyro_df[(gyro_df['User'] == user) & (gyro_df['Device'] == device)].copy()
            
            if len(a) < window_size or len(g) < window_size:
                continue
            
            a = a.sort_values('Creation_Time').reset_index(drop=True)
            g = g.sort_values('Creation_Time').reset_index(drop=True)
            
            min_len = min(len(a), len(g))
            a = a.iloc[:min_len]
            g = g.iloc[:min_len]
            
            sensor_data = np.column_stack([
                a[['x', 'y', 'z']].values.astype(np.float32),
                g[['x', 'y', 'z']].values.astype(np.float32)
            ])
            
            labels_str = a['gt'].values
            labels = np.array([HHAR_ACTIVITY_MAP.get(l, -1) for l in labels_str])
            
            nan_mask = np.isnan(sensor_data).any(axis=1)
            if nan_mask.any():
                sensor_df = pd.DataFrame(sensor_data)
                sensor_df = clean_nan_sensor_data(sensor_df)
                sensor_data = sensor_df.values.astype(np.float32)
            
            for start in range(0, len(sensor_data) - window_size + 1, step_size):
                end = start + window_size
                seg = sensor_data[start:end]
                seg_labels = labels[start:end]
                
                if (seg_labels == -1).any():
                    continue
                label = np.bincount(seg_labels[seg_labels >= 0]).argmax()
                
                if user in test_users:
                    test_windows.append(seg)
                    test_labels.append(label)
                else:
                    train_windows.append(seg)
                    train_labels.append(label)
    
    X_train = np.stack(train_windows)
    y_train = np.array(train_labels)
    X_test = np.stack(test_windows)
    y_test = np.array(test_labels)
    
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"[HHAR] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Features: {X_train.shape[2]}, Classes: {len(set(y_train))}")
    return X_train, y_train, X_test, y_test


def compute_frequency_dominance(X, ratio_threshold=1.0):
    N, seq_len, channels = X.shape
    labels = []
    ratios = []
    
    half = seq_len // 2
    low_end = half // 2   # 저주파: DC ~ 1/4 Nyquist
    
    for i in range(N):
        total_high_energy = 0.0
        total_low_energy = 0.0
        
        for ch in range(channels):
            signal = X[i, :, ch]
            fft_vals = np.fft.rfft(signal)
            magnitude = np.abs(fft_vals)
            
            low_energy = np.sum(magnitude[1:low_end+1] ** 2)
            high_energy = np.sum(magnitude[low_end+1:] ** 2)
            
            total_low_energy += low_energy
            total_high_energy += high_energy
        
        ratio = total_high_energy / (total_low_energy + 1e-10)
        ratios.append(ratio)
        labels.append('high' if ratio >= ratio_threshold else 'low')
    
    return np.array(labels), np.array(ratios)