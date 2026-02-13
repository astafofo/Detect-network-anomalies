#!/usr/bin/env python3
"""
Example usage of the Network Anomaly Detection System

This script demonstrates various ways to use the anomaly detection pipeline
with different configurations and scenarios.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import NetworkAnomalyDetectionPipeline
from src.data_generator import NetworkDataGenerator
from src.feature_extractor import NetworkFeatureExtractor
from src.models import IsolationForestModel, OneClassSVMModel, AutoencoderModel
from src.evaluator import AnomalyEvaluator


def example_1_basic_pipeline():
    """Example 1: Basic pipeline with synthetic data"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline with Synthetic Data")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = NetworkAnomalyDetectionPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print(f"Best model performance: {max(results['evaluation_results'].items(), key=lambda x: x[1]['roc_auc'])}")


def example_2_custom_configuration():
    """Example 2: Custom configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration
    config = {
        'data_generation': {
            'n_samples': 5000,
            'anomaly_ratio': 0.1,
            'seed': 123
        },
        'models': {
            'isolation_forest': {
                'contamination': 0.15,
                'random_state': 123
            },
            'autoencoder': {
                'encoding_dim': 16,
                'learning_rate': 0.01,
                'epochs': 50
            }
        }
    }
    
    # Initialize with custom config
    pipeline = NetworkAnomalyDetectionPipeline(config)
    
    # Generate data
    data, metadata = pipeline.generate_data()
    
    # Extract features
    X, processed_data = pipeline.extract_features()
    
    # Split data
    X_train, X_test, y_train, y_test = pipeline.split_data()
    
    # Train only specific models
    pipeline.train_models(['isolation_forest', 'autoencoder'])
    
    # Evaluate
    results = pipeline.evaluate_models()
    
    for model_name, metrics in results.items():
        print(f"{model_name}: ROC-AUC = {metrics['roc_auc']:.4f}")


def example_3_individual_model_training():
    """Example 3: Training individual models"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Individual Model Training")
    print("=" * 60)
    
    # Generate data
    generator = NetworkDataGenerator()
    data, metadata = generator.generate_dataset(n_samples=3000, anomaly_ratio=0.08)
    
    # Extract features
    extractor = NetworkFeatureExtractor()
    X, processed_data = extractor.fit_transform(data)
    y = processed_data['is_anomaly'].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForestModel(contamination=0.1)
    iso_forest.fit(X_train, y_train)
    
    # Train Autoencoder
    print("Training Autoencoder...")
    autoencoder = AutoencoderModel(
        input_dim=X_train.shape[1],
        encoding_dim=16,
        epochs=50
    )
    autoencoder.fit(X_train, y_train)
    
    # Evaluate both models
    evaluator = AnomalyEvaluator()
    
    for name, model in [("Isolation Forest", iso_forest), ("Autoencoder", autoencoder)]:
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)
        
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_scores)
        print(f"{name}:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  - F1-Score: {metrics['f1_score']:.4f}")


def example_4_real_time_detection():
    """Example 4: Real-time anomaly detection simulation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Real-time Detection Simulation")
    print("=" * 60)
    
    # Train a model first
    pipeline = NetworkAnomalyDetectionPipeline()
    pipeline.generate_data(n_samples=5000, anomaly_ratio=0.05)
    pipeline.extract_features()
    pipeline.split_data()
    pipeline.train_models(['isolation_forest'])
    
    # Simulate real-time data
    print("Simulating real-time network traffic...")
    
    # Generate new data (simulating live traffic)
    new_data, _ = pipeline.generate_data(n_samples=100, anomaly_ratio=0.1)
    
    # Detect anomalies
    y_pred, y_scores, annotated_data = pipeline.detect_anomalies(new_data)
    
    # Show results
    anomalies_detected = y_pred.sum()
    print(f"Processed {len(new_data)} new samples")
    print(f"Detected {anomalies_detected} anomalies")
    
    if anomalies_detected > 0:
        print("\nDetected anomalies:")
        anomaly_data = annotated_data[annotated_data['predicted_anomaly'] == 1]
        for idx, row in anomaly_data.head(5).iterrows():
            print(f"  - Score: {row['anomaly_score']:.4f}, "
                  f"Src: {row['src_ip']}, Dst: {row['dst_ip']}, "
                  f"Port: {row['dst_port']}")


def example_5_custom_data():
    """Example 5: Using custom CSV data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Data Processing")
    print("=" * 60)
    
    # Create sample custom data
    print("Creating sample network data...")
    
    # Generate sample data
    generator = NetworkDataGenerator(seed=456)
    sample_data, _ = generator.generate_dataset(n_samples=2000, anomaly_ratio=0.07)
    
    # Save to CSV
    sample_data.to_csv('sample_network_data.csv', index=False)
    print("Sample data saved to 'sample_network_data.csv'")
    
    # Load and process with pipeline
    pipeline = NetworkAnomalyDetectionPipeline()
    pipeline.load_data('sample_network_data.csv')
    pipeline.extract_features()
    pipeline.split_data()
    pipeline.train_models(['isolation_forest', 'one_class_svm'])
    results = pipeline.evaluate_models()
    
    print("Results on custom data:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: ROC-AUC = {metrics['roc_auc']:.4f}")
    
    # Clean up
    os.remove('sample_network_data.csv')


def example_6_visualization():
    """Example 6: Creating visualizations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Visualization Dashboard")
    print("=" * 60)
    
    # Train models
    pipeline = NetworkAnomalyDetectionPipeline()
    pipeline.generate_data(n_samples=3000, anomaly_ratio=0.08)
    pipeline.extract_features()
    pipeline.split_data()
    pipeline.train_models(['isolation_forest', 'autoencoder'])
    
    # Get predictions for visualization
    model = pipeline.trained_models['isolation_forest']
    y_pred = model.predict(pipeline.X_test)
    y_scores = model.predict_proba(pipeline.X_test)
    
    # Create visualizations
    evaluator = AnomalyEvaluator()
    
    # Create dashboard
    dashboard = evaluator.create_dashboard(
        pipeline.y_test, y_pred, y_scores,
        model_name="Isolation Forest",
        timestamps=pipeline.processed_data['timestamp'].iloc[-len(pipeline.y_test):],
        feature_names=pipeline.feature_extractor.get_feature_names()
    )
    
    # Save visualizations
    os.makedirs('visualizations', exist_ok=True)
    for name, fig in dashboard.items():
        fig.write_html(f'visualizations/{name}.html')
    
    print("Visualizations saved to 'visualizations/' directory:")
    for name in dashboard.keys():
        print(f"  - {name}.html")


def main():
    """Run all examples"""
    print("Network Anomaly Detection System - Examples")
    print("==========================================")
    
    try:
        # Run all examples
        example_1_basic_pipeline()
        example_2_custom_configuration()
        example_3_individual_model_training()
        example_4_real_time_detection()
        example_5_custom_data()
        example_6_visualization()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
