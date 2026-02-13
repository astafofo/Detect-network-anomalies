import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import ipaddress


class NetworkFeatureExtractor:
    """Extract and engineer features from network traffic data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from IP addresses."""
        df = df.copy()
        
        # IP address numerical features
        def ip_to_int(ip_str):
            try:
                return int(ipaddress.IPv4Address(ip_str))
            except:
                return 0
        
        df['src_ip_int'] = df['src_ip'].apply(ip_to_int)
        df['dst_ip_int'] = df['dst_ip'].apply(ip_to_int)
        
        # IP class features
        def get_ip_class(ip_str):
            try:
                ip = ipaddress.IPv4Address(ip_str)
                if ip.is_private:
                    return 'private'
                elif ip.is_loopback:
                    return 'loopback'
                elif ip.is_multicast:
                    return 'multicast'
                else:
                    return 'public'
            except:
                return 'unknown'
        
        df['src_ip_class'] = df['src_ip'].apply(get_ip_class)
        df['dst_ip_class'] = df['dst_ip'].apply(get_ip_class)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            df['is_business_hours'] = ((df['timestamp'].dt.hour >= 9) & 
                                      (df['timestamp'].dt.hour <= 17)).astype(int)
        
        return df
    
    def extract_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract traffic pattern features."""
        df = df.copy()
        
        # Ratio features
        df['bytes_per_packet_sent'] = df['bytes_sent'] / (df['packets_sent'] + 1)
        df['bytes_per_packet_received'] = df['bytes_received'] / (df['packets_received'] + 1)
        df['packet_ratio'] = df['packets_sent'] / (df['packets_received'] + 1)
        df['byte_ratio'] = df['bytes_sent'] / (df['bytes_received'] + 1)
        
        # Port features
        df['is_well_known_port'] = (df['dst_port'] < 1024).astype(int)
        df['is_ephemeral_port'] = (df['src_port'] >= 32768).astype(int)
        
        # Packet size features
        df['is_large_packet'] = (df['packet_size'] > 1500).astype(int)
        df['is_small_packet'] = (df['packet_size'] < 64).astype(int)
        
        # Protocol-specific features
        df['is_tcp'] = (df['protocol'] == 'TCP').astype(int)
        df['is_udp'] = (df['protocol'] == 'UDP').astype(int)
        df['is_icmp'] = (df['protocol'] == 'ICMP').astype(int)
        
        return df
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features for rolling windows."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
            # Rolling statistics (window of 10 packets)
            window_size = 10
            
            for col in ['packet_size', 'duration', 'bytes_sent', 'bytes_received']:
                if col in df.columns:
                    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                    df[f'{col}_rolling_max'] = df[col].rolling(window=window_size, min_periods=1).max()
                    df[f'{col}_rolling_min'] = df[col].rolling(window=window_size, min_periods=1).min()
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        categorical_columns = ['protocol', 'flags', 'src_ip_class', 'dst_ip_class']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        unique_values = set(df[col].astype(str).unique())
                        known_values = set(self.label_encoders[col].classes_)
                        
                        # Replace unseen values with most common known value
                        unseen_values = unique_values - known_values
                        if unseen_values:
                            most_common = self.label_encoders[col].classes_[0]
                            df[col] = df[col].astype(str).replace(list(unseen_values), most_common)
                        
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fit the extractor and transform the data."""
        
        # Extract all features
        df_processed = df.copy()
        df_processed = self.extract_ip_features(df_processed)
        df_processed = self.extract_temporal_features(df_processed)
        df_processed = self.extract_traffic_features(df_processed)
        df_processed = self.extract_statistical_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=True)
        
        # Select numerical features for modeling
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_columns = ['timestamp', 'src_ip', 'dst_ip', 'is_anomaly']
        feature_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        self.feature_columns = feature_columns
        
        # Handle missing values
        df_processed[feature_columns] = df_processed[feature_columns].fillna(0)
        
        # Scale features
        X = df_processed[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.is_fitted = True
        
        return X_scaled, df_processed
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Transform new data using fitted extractor."""
        
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transforming data")
        
        # Extract all features
        df_processed = df.copy()
        df_processed = self.extract_ip_features(df_processed)
        df_processed = self.extract_temporal_features(df_processed)
        df_processed = self.extract_traffic_features(df_processed)
        df_processed = self.extract_statistical_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=False)
        
        # Select feature columns
        feature_columns = [col for col in self.feature_columns if col in df_processed.columns]
        
        # Handle missing values
        df_processed[feature_columns] = df_processed[feature_columns].fillna(0)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Scale features
        X = df_processed[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, df_processed
    
    def get_feature_names(self) -> list:
        """Get the names of the extracted features."""
        return self.feature_columns.copy()
