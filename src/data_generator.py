import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, Any


class NetworkDataGenerator:
    """Generate synthetic network traffic data for anomaly detection."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_normal_traffic(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate normal network traffic patterns."""
        
        # Time-based features
        start_time = datetime.now() - timedelta(hours=24)
        timestamps = [start_time + timedelta(seconds=i*10) for i in range(n_samples)]
        
        # Normal network traffic characteristics
        data = {
            'timestamp': timestamps,
            'src_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'dst_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.choice([80, 443, 22, 53, 25, 110, 143], n_samples),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.7, 0.25, 0.05]),
            'packet_size': np.random.lognormal(7, 0.5, n_samples).astype(int),
            'duration': np.random.exponential(0.1, n_samples),
            'flags': np.random.choice(['SYN', 'ACK', 'FIN', 'RST'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'bytes_sent': np.random.lognormal(10, 1, n_samples).astype(int),
            'bytes_received': np.random.lognormal(10, 1, n_samples).astype(int),
            'packets_sent': np.random.poisson(5, n_samples),
            'packets_received': np.random.poisson(5, n_samples),
            'is_anomaly': np.zeros(n_samples, dtype=int)
        }
        
        # Add some correlations
        data['packet_size'] = np.where(data['protocol'] == 'ICMP', 
                                      np.random.normal(64, 10, n_samples).astype(int),
                                      data['packet_size'])
        
        return pd.DataFrame(data)
    
    def generate_anomalies(self, normal_data: pd.DataFrame, anomaly_ratio: float = 0.05) -> pd.DataFrame:
        """Inject various types of anomalies into normal traffic."""
        
        n_anomalies = int(len(normal_data) * anomaly_ratio)
        anomaly_data = normal_data.sample(n=n_anomalies, random_state=42).copy()
        
        anomaly_types = ['ddos', 'port_scan', 'data_exfiltration', 'unusual_protocol']
        
        for i in range(n_anomalies):
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'ddos':
                # High volume traffic from single source
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('src_ip')] = "192.168.1.100"
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('packets_sent')] = np.random.poisson(1000)
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('bytes_sent')] = np.random.poisson(100000)
                
            elif anomaly_type == 'port_scan':
                # Multiple destination ports
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('dst_port')] = np.random.randint(1, 65535)
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('flags')] = 'SYN'
                
            elif anomaly_type == 'data_exfiltration':
                # Large outbound data transfers
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('bytes_sent')] = np.random.poisson(1000000)
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('packet_size')] = np.random.normal(1500, 100, 1)[0]
                
            elif anomaly_type == 'unusual_protocol':
                # Unusual protocol usage
                anomaly_data.iloc[i, anomaly_data.columns.get_loc('protocol')] = np.random.choice(['GRE', 'ESP', 'AH'])
        
        anomaly_data['is_anomaly'] = 1
        
        # Combine normal and anomalous data
        combined_data = pd.concat([normal_data, anomaly_data], ignore_index=True)
        return combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def generate_dataset(self, n_samples: int = 10000, anomaly_ratio: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate complete dataset with normal and anomalous traffic."""
        
        normal_data = self.generate_normal_traffic(int(n_samples * (1 - anomaly_ratio)))
        full_data = self.generate_anomalies(normal_data, anomaly_ratio)
        
        metadata = {
            'total_samples': len(full_data),
            'normal_samples': len(full_data) - full_data['is_anomaly'].sum(),
            'anomaly_samples': full_data['is_anomaly'].sum(),
            'anomaly_ratio': full_data['is_anomaly'].mean(),
            'features': list(full_data.columns)
        }
        
        return full_data, metadata
