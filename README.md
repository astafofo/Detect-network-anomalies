# Network Anomaly Detection System

A comprehensive machine learning system for detecting network anomalies using multiple algorithms including Isolation Forest, One-Class SVM, and Autoencoders.

## Features

- **Multiple ML Models**: Isolation Forest, One-Class SVM, and Autoencoder-based anomaly detection
- **Ensemble Methods**: Combines multiple models for improved performance
- **Feature Engineering**: Automatic extraction of network traffic features
- **Data Generation**: Synthetic network traffic generation for testing
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Real-time Detection**: Pipeline for detecting anomalies in live network data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/astafofo/Detect-network-anomalies-using-machine-learning-models.git
cd "Detect network anomalies"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run the complete pipeline with synthetic data:
```bash
python main.py --demo
```

This will:
- Generate 10,000 samples of network traffic with 5% anomalies
- Train all three models (Isolation Forest, One-Class SVM, Autoencoder)
- Create an ensemble model
- Generate evaluation reports and visualizations
- Save results to the `results/` directory

### Custom Parameters

Generate custom dataset:
```bash
python main.py --samples 50000 --anomaly-ratio 0.1 --demo
```

Use specific models:
```bash
python main.py --models isolation_forest autoencoder --demo
```

### Using Your Own Data

Prepare your CSV file with the following columns:
- `timestamp`: Timestamp of the network event
- `src_ip`: Source IP address
- `dst_ip`: Destination IP address  
- `src_port`: Source port
- `dst_port`: Destination port
- `protocol`: Protocol (TCP, UDP, ICMP)
- `packet_size`: Size of packets in bytes
- `duration`: Connection duration
- `flags`: TCP flags
- `bytes_sent`: Bytes sent
- `bytes_received`: Bytes received
- `packets_sent`: Number of packets sent
- `packets_received`: Number of packets received
- `is_anomaly`: Ground truth label (optional, 0=normal, 1=anomaly)

Run with your data:
```bash
python main.py --data your_network_data.csv --demo
```

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic network traffic generation
│   ├── feature_extractor.py   # Feature engineering and preprocessing
│   ├── models.py             # ML models (Isolation Forest, SVM, Autoencoder)
│   ├── evaluator.py          # Evaluation metrics and visualizations
│   └── pipeline.py           # Complete detection pipeline
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Models

### 1. Isolation Forest
- Unsupervised tree-based anomaly detection
- Fast and efficient for large datasets
- Good for high-dimensional data

### 2. One-Class SVM
- Kernel-based anomaly detection
- Effective for complex patterns
- Works well with non-linear data

### 3. Autoencoder
- Neural network-based reconstruction
- Learns normal patterns and detects deviations
- Captures complex non-linear relationships

### 4. Ensemble Model
- Combines predictions from all models
- Improves robustness and accuracy
- Uses voting mechanism for final predictions

## Features

The system automatically extracts the following features:

### Basic Features
- IP address numerical representations
- Port number features
- Protocol encoding
- Packet statistics

### Temporal Features
- Hour of day
- Day of week
- Weekend/business hours indicators

### Traffic Features
- Bytes per packet ratios
- Packet size classifications
- Port type indicators
- Protocol-specific features

### Statistical Features
- Rolling window statistics
- Moving averages and standard deviations
- Min/max values over time windows

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Detection sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

## Visualizations

When using the `--demo` flag, the system generates:

- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Anomaly Score Distributions
- Timeline Plots
- Feature Importance Charts

## Configuration

You can customize the pipeline by modifying the configuration in `main.py` or by creating a config dictionary:

```python
config = {
    'data_generation': {
        'n_samples': 10000,
        'anomaly_ratio': 0.05,
        'seed': 42
    },
    'models': {
        'isolation_forest': {
            'contamination': 0.1,
            'random_state': 42
        },
        'autoencoder': {
            'encoding_dim': 32,
            'epochs': 100
        }
    }
}

pipeline = NetworkAnomalyDetectionPipeline(config)
```

## API Usage

You can also use the pipeline programmatically:

```python
from src.pipeline import NetworkAnomalyDetectionPipeline

# Initialize pipeline
pipeline = NetworkAnomalyDetectionPipeline()

# Generate synthetic data
data, metadata = pipeline.generate_data(n_samples=10000, anomaly_ratio=0.05)

# Extract features
X, processed_data = pipeline.extract_features()

# Split data
X_train, X_test, y_train, y_test = pipeline.split_data()

# Train models
pipeline.train_models(['isolation_forest', 'autoencoder'])

# Evaluate
results = pipeline.evaluate_models()

# Detect anomalies on new data
y_pred, y_scores, annotated_data = pipeline.detect_anomalies()
```

## Performance Tips

1. **Large Datasets**: Use Isolation Forest for faster training
2. **Complex Patterns**: Use Autoencoder for non-linear relationships
3. **Imbalanced Data**: Adjust `contamination` parameter in Isolation Forest
4. **Real-time Detection**: Pre-train models and use `detect_anomalies()` method

## Saving and Loading Models

The system automatically saves trained models to the `models/` directory:

```python
# Save models
pipeline.save_models('my_models')

# Load models later
pipeline.load_models('my_models')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Requirements

- Python 3.7+
- TensorFlow 2.x (for Autoencoder)
- scikit-learn
- pandas
- numpy
- plotly (for visualizations)
- matplotlib
- seaborn

See `requirements.txt` for exact versions.
