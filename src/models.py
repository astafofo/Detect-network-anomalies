import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, Tuple, Any
import joblib


class AnomalyDetectionModel:
    """Base class for anomaly detection models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the model to training data."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)."""
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        raise NotImplementedError
        
    def save_model(self, filepath: str):
        """Save the trained model."""
        raise NotImplementedError
        
    def load_model(self, filepath: str):
        """Load a trained model."""
        raise NotImplementedError


class IsolationForestModel(AnomalyDetectionModel):
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__("IsolationForest")
        self.contamination = contamination
        self.random_state = random_state
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit Isolation Forest model."""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.model.fit(X)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        # Convert to binary (1 for anomaly, 0 for normal)
        return (predictions == -1).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(X)
        # Convert to probability-like scores (0-1, higher = more anomalous)
        min_score, max_score = scores.min(), scores.max()
        if max_score - min_score > 0:
            normalized_scores = (max_score - scores) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)
        
        return normalized_scores
        
    def save_model(self, filepath: str):
        """Save the model."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            
    def load_model(self, filepath: str):
        """Load the model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True


class OneClassSVMModel(AnomalyDetectionModel):
    """One-Class SVM for anomaly detection."""
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf'):
        super().__init__("OneClassSVM")
        self.nu = nu
        self.kernel = kernel
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit One-Class SVM model."""
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel)
        self.model.fit(X)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        scores = self.model.decision_function(X)
        # Convert to probability-like scores
        min_score, max_score = scores.min(), scores.max()
        if max_score - min_score > 0:
            normalized_scores = (max_score - scores) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)
        
        return normalized_scores
        
    def save_model(self, filepath: str):
        """Save the model."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            
    def load_model(self, filepath: str):
        """Load the model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True


class AutoencoderModel(AnomalyDetectionModel):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 learning_rate: float = 0.001, epochs: int = 100, batch_size: int = 32):
        super().__init__("Autoencoder")
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = None
        
    def _build_model(self):
        """Build the autoencoder architecture."""
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(self.encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(self.encoding_dim * 2, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return autoencoder
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the autoencoder model."""
        self.model = self._build_model()
        
        # Train on normal data only (assuming y=0 for normal)
        if y is not None:
            X_normal = X[y == 0]
        else:
            X_normal = X
            
        # Split for validation
        X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
        
        # Train the model
        history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, X_val),
            verbose=0
        )
        
        # Calculate reconstruction error threshold using validation data
        val_predictions = self.model.predict(X_val)
        mse = np.mean(np.power(X_val - val_predictions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
        
        self.is_fitted = True
        return history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        return (mse > self.threshold).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        
        # Normalize MSE to probability-like scores
        min_mse, max_mse = mse.min(), mse.max()
        if max_mse - min_mse > 0:
            normalized_scores = (mse - min_mse) / (max_mse - min_mse)
        else:
            normalized_scores = np.zeros_like(mse)
            
        return normalized_scores
        
    def save_model(self, filepath: str):
        """Save the model."""
        if self.model is not None:
            self.model.save(filepath)
            
    def load_model(self, filepath: str):
        """Load the model."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True


class EnsembleModel:
    """Ensemble of multiple anomaly detection models."""
    
    def __init__(self, models: list):
        self.models = models
        self.weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble voting."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores using ensemble."""
        scores = []
        for model in self.models:
            score = model.predict_proba(X)
            scores.append(score)
        
        # Average scores
        ensemble_score = np.mean(scores, axis=0)
        return ensemble_score
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble and individual models."""
        results = {}
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X)
        ensemble_scores = self.predict_proba(X)
        
        results['ensemble'] = {
            'predictions': ensemble_pred,
            'scores': ensemble_scores,
            'classification_report': classification_report(y, ensemble_pred, output_dict=True),
            'auc_roc': roc_auc_score(y, ensemble_scores)
        }
        
        # Evaluate individual models
        for model in self.models:
            pred = model.predict(X)
            scores = model.predict_proba(X)
            
            results[model.model_name] = {
                'predictions': pred,
                'scores': scores,
                'classification_report': classification_report(y, pred, output_dict=True),
                'auc_roc': roc_auc_score(y, scores)
            }
            
        return results
