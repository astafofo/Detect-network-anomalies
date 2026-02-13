import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import os
import json
from datetime import datetime

from .data_generator import NetworkDataGenerator
from .feature_extractor import NetworkFeatureExtractor
from .models import IsolationForestModel, OneClassSVMModel, AutoencoderModel, EnsembleModel
from .evaluator import AnomalyEvaluator


class NetworkAnomalyDetectionPipeline:
    """Complete pipeline for network anomaly detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline with configuration."""
        
        # Default configuration
        self.config = {
            'data_generation': {
                'n_samples': 10000,
                'anomaly_ratio': 0.05,
                'seed': 42
            },
            'feature_extraction': {
                'use_rolling_features': True,
                'rolling_window_size': 10
            },
            'models': {
                'isolation_forest': {
                    'contamination': 0.1,
                    'random_state': 42
                },
                'one_class_svm': {
                    'nu': 0.1,
                    'kernel': 'rbf'
                },
                'autoencoder': {
                    'encoding_dim': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32
                }
            },
            'evaluation': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
        
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self.data_generator = NetworkDataGenerator(seed=self.config['data_generation']['seed'])
        self.feature_extractor = NetworkFeatureExtractor()
        self.evaluator = AnomalyEvaluator()
        
        # Models
        self.models = {}
        self.ensemble = None
        
        # Data
        self.raw_data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results
        self.results = {}
        self.trained_models = {}
        
    def _update_config(self, base_config: Dict, new_config: Dict):
        """Recursively update configuration."""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def generate_data(self, n_samples: int = None, anomaly_ratio: float = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic network traffic data."""
        
        n_samples = n_samples or self.config['data_generation']['n_samples']
        anomaly_ratio = anomaly_ratio or self.config['data_generation']['anomaly_ratio']
        
        self.raw_data, metadata = self.data_generator.generate_dataset(n_samples, anomaly_ratio)
        
        print(f"Generated dataset:")
        print(f"  - Total samples: {metadata['total_samples']}")
        print(f"  - Normal samples: {metadata['normal_samples']}")
        print(f"  - Anomaly samples: {metadata['anomaly_samples']}")
        print(f"  - Anomaly ratio: {metadata['anomaly_ratio']:.3f}")
        
        return self.raw_data, metadata
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        
        self.raw_data = pd.read_csv(filepath)
        
        if 'timestamp' in self.raw_data.columns:
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
        
        print(f"Loaded dataset with {len(self.raw_data)} samples")
        return self.raw_data
    
    def extract_features(self, data: pd.DataFrame = None, fit: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extract features from network data."""
        
        if data is None:
            data = self.raw_data
        
        if data is None:
            raise ValueError("No data available. Generate or load data first.")
        
        if fit:
            self.X, self.processed_data = self.feature_extractor.fit_transform(data)
            self.y = self.processed_data['is_anomaly'].values if 'is_anomaly' in self.processed_data.columns else None
        else:
            self.X, self.processed_data = self.feature_extractor.transform(data)
            self.y = self.processed_data['is_anomaly'].values if 'is_anomaly' in self.processed_data.columns else None
        
        print(f"Extracted {self.X.shape[1]} features from {len(self.X)} samples")
        
        return self.X, self.processed_data
    
    def split_data(self, test_size: float = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        
        if self.X is None:
            raise ValueError("No features extracted. Run extract_features() first.")
        
        test_size = test_size or self.config['evaluation']['test_size']
        random_state = random_state or self.config['evaluation']['random_state']
        
        from sklearn.model_selection import train_test_split
        
        if self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
        else:
            self.X_train, self.X_test = train_test_split(
                self.X, test_size=test_size, random_state=random_state
            )
            self.y_train = None
            self.y_test = None
        
        print(f"Split data:")
        print(f"  - Training: {len(self.X_train)} samples")
        print(f"  - Testing: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self, model_names: list = None) -> Dict[str, Any]:
        """Train anomaly detection models."""
        
        if self.X_train is None:
            raise ValueError("No training data available. Run split_data() first.")
        
        if model_names is None:
            model_names = ['isolation_forest', 'one_class_svm', 'autoencoder']
        
        input_dim = self.X_train.shape[1]
        
        # Initialize models
        if 'isolation_forest' in model_names:
            config = self.config['models']['isolation_forest']
            self.models['isolation_forest'] = IsolationForestModel(
                contamination=config['contamination'],
                random_state=config['random_state']
            )
        
        if 'one_class_svm' in model_names:
            config = self.config['models']['one_class_svm']
            self.models['one_class_svm'] = OneClassSVMModel(
                nu=config['nu'],
                kernel=config['kernel']
            )
        
        if 'autoencoder' in model_names:
            config = self.config['models']['autoencoder']
            self.models['autoencoder'] = AutoencoderModel(
                input_dim=input_dim,
                encoding_dim=config['encoding_dim'],
                learning_rate=config['learning_rate'],
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
        
        # Train models
        training_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'autoencoder' and hasattr(model, 'fit'):
                history = model.fit(self.X_train, self.y_train)
                training_results[name] = {'training_history': history}
            else:
                model.fit(self.X_train, self.y_train)
                training_results[name] = {'status': 'completed'}
            
            self.trained_models[name] = model
            print(f"  ✓ {name} trained successfully")
        
        # Create ensemble
        if len(self.trained_models) > 1:
            self.ensemble = EnsembleModel(list(self.trained_models.values()))
            self.ensemble.fit(self.X_train, self.y_train)
            print("  ✓ Ensemble model created")
        
        return training_results
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models."""
        
        if self.X_test is None:
            raise ValueError("No test data available. Run split_data() first.")
        
        if not self.trained_models:
            raise ValueError("No trained models available. Run train_models() first.")
        
        evaluation_results = {}
        
        # Evaluate individual models
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            
            y_pred = model.predict(self.X_test)
            y_scores = model.predict_proba(self.X_test)
            
            metrics = self.evaluator.calculate_metrics(self.y_test, y_pred, y_scores)
            evaluation_results[name] = metrics
            
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Evaluate ensemble
        if self.ensemble:
            print("Evaluating ensemble...")
            
            y_pred = self.ensemble.predict(self.X_test)
            y_scores = self.ensemble.predict_proba(self.X_test)
            
            metrics = self.evaluator.calculate_metrics(self.y_test, y_pred, y_scores)
            evaluation_results['ensemble'] = metrics
            
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.results = evaluation_results
        return evaluation_results
    
    def detect_anomalies(self, data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Detect anomalies in new data."""
        
        if not self.trained_models:
            raise ValueError("No trained models available. Run train_models() first.")
        
        # Use ensemble if available, otherwise use the first trained model
        model = self.ensemble if self.ensemble else list(self.trained_models.values())[0]
        
        # Extract features
        if data is None:
            X = self.X
            processed_data = self.processed_data
        else:
            X, processed_data = self.feature_extractor.transform(data)
        
        # Make predictions
        y_pred = model.predict(X)
        y_scores = model.predict_proba(X)
        
        # Add predictions to processed data
        processed_data['predicted_anomaly'] = y_pred
        processed_data['anomaly_score'] = y_scores
        
        return y_pred, y_scores, processed_data
    
    def save_models(self, directory: str = "models"):
        """Save trained models to disk."""
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.trained_models.items():
            filepath = os.path.join(directory, f"{name}.pkl")
            model.save_model(filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save feature extractor
        import joblib
        joblib.dump(self.feature_extractor, os.path.join(directory, "feature_extractor.pkl"))
        
        # Save configuration
        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Saved all models and configuration to {directory}")
    
    def load_models(self, directory: str = "models"):
        """Load trained models from disk."""
        
        import joblib
        
        # Load feature extractor
        self.feature_extractor = joblib.load(os.path.join(directory, "feature_extractor.pkl"))
        
        # Load configuration
        with open(os.path.join(directory, "config.json"), 'r') as f:
            self.config = json.load(f)
        
        # Load models
        model_files = {
            'isolation_forest': IsolationForestModel(),
            'one_class_svm': OneClassSVMModel(),
            'autoencoder': AutoencoderModel(input_dim=1)  # Will be updated when loading
        }
        
        for name, model_class in model_files.items():
            filepath = os.path.join(directory, f"{name}.pkl")
            if os.path.exists(filepath):
                model = model_class
                model.load_model(filepath)
                self.trained_models[name] = model
                print(f"Loaded {name} from {filepath}")
        
        # Create ensemble
        if len(self.trained_models) > 1:
            self.ensemble = EnsembleModel(list(self.trained_models.values()))
        
        print(f"Loaded all models from {directory}")
    
    def run_complete_pipeline(self, data_path: str = None, save_models: bool = True) -> Dict[str, Any]:
        """Run the complete anomaly detection pipeline."""
        
        print("=" * 60)
        print("NETWORK ANOMALY DETECTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load or generate data
        if data_path:
            print("\n1. Loading data...")
            self.load_data(data_path)
        else:
            print("\n1. Generating synthetic data...")
            self.generate_data()
        
        # Step 2: Extract features
        print("\n2. Extracting features...")
        self.extract_features()
        
        # Step 3: Split data
        print("\n3. Splitting data...")
        self.split_data()
        
        # Step 4: Train models
        print("\n4. Training models...")
        training_results = self.train_models()
        
        # Step 5: Evaluate models
        print("\n5. Evaluating models...")
        evaluation_results = self.evaluate_models()
        
        # Step 6: Save models
        if save_models:
            print("\n6. Saving models...")
            self.save_models()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'config': self.config
        }
