"""
ICU Sepsis Data Preprocessing Module
Handles loading, cleaning, normalization, and sequence generation
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


class SepsisDataProcessor:
    """Preprocessing pipeline for ICU sepsis time-series data"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.features = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'WBC', 'Lactate']
        self.target = 'SepsisLabel'
        self.id_col = 'patient_id'
        self.scaler = MinMaxScaler()

    def load_and_merge_psv(self, directory_path: str) -> pd.DataFrame:
        """Load all .psv files from directory and merge"""
        if not os.path.exists(directory_path):
            return pd.DataFrame()

        all_dfs = []
        files = [f for f in os.listdir(directory_path) if f.endswith('.psv')]
        
        for file in files:
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path, sep='|')
            df[self.id_col] = file.split('.')[0]
            all_dfs.append(df)
            
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and select features"""
        cols = [self.id_col] + [f for f in self.features if f in df.columns]
        if self.target in df.columns:
            cols.append(self.target)
        df = df[cols].copy()

        df[self.features] = df.groupby(self.id_col)[self.features].ffill()
        df[self.features] = df[self.features].fillna(0)
        
        return df

    def normalize(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize features using MinMaxScaler"""
        self.scaler.fit(train_df[self.features])
        train_df[self.features] = self.scaler.transform(train_df[self.features])
        test_df[self.features] = self.scaler.transform(test_df[self.features])
        return train_df, test_df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame into sliding window sequences for LSTM"""
        if len(df) < self.window_size:
            return np.array([]), np.array([])

        feat_values = df[self.features].values
        patient_ids = df[self.id_col].values
        
        windows = np.lib.stride_tricks.sliding_window_view(feat_values, (self.window_size, len(self.features)))
        X_all = windows.reshape(-1, self.window_size, len(self.features))
        
        pid_windows = np.lib.stride_tricks.sliding_window_view(patient_ids, self.window_size)
        valid_mask = pid_windows[:, 0] == pid_windows[:, -1]
        
        X = X_all[valid_mask]
        
        if self.target in df.columns:
            target_values = df[self.target].values
            y_windows = np.lib.stride_tricks.sliding_window_view(target_values, self.window_size)
            y = y_windows[valid_mask][:, -1]
            return X, y
            
        return X, np.array([])

    def run(self, data_dir: str = "./data/training", fallback_csv: str = None):
        """Run complete preprocessing pipeline"""
        raw_df = self.load_and_merge_psv(data_dir)
        
        if raw_df.empty and fallback_csv and os.path.exists(fallback_csv):
            raw_df = pd.read_csv(fallback_csv)
            if 'Patient_ID' in raw_df.columns:
                raw_df = raw_df.rename(columns={'Patient_ID': 'patient_id'})
        
        if raw_df.empty:
            raise ValueError("No data found")

        processed_df = self.process_pipeline(raw_df)
        
        patient_ids = processed_df[self.id_col].unique()
        train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
        
        train_df = processed_df[processed_df[self.id_col].isin(train_ids)].copy()
        test_df = processed_df[processed_df[self.id_col].isin(test_ids)].copy()
        
        train_df, test_df = self.normalize(train_df, test_df)
        
        X_train, y_train = self.create_sequences(train_df)
        X_test, y_test = self.create_sequences(test_df)
        
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)
        
        return X_train, y_train, X_test, y_test
