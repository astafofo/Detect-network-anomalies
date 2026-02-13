#!/usr/bin/env python3
"""
Network Anomaly Detection System

This script demonstrates a complete network anomaly detection pipeline using
machine learning models including Isolation Forest, One-Class SVM, and Autoencoders.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import NetworkAnomalyDetectionPipeline
from src.evaluator import AnomalyEvaluator


def main():
    parser = argparse.ArgumentParser(description='Network Anomaly Detection System')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--anomaly-ratio', type=float, default=0.05, help='Anomaly ratio in data')
    parser.add_argument('--models', nargs='+', 
                       choices=['isolation_forest', 'one_class_svm', 'autoencoder'],
                       default=['isolation_forest', 'one_class_svm', 'autoencoder'],
                       help='Models to train')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save models')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demo with visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing Network Anomaly Detection Pipeline...")
    pipeline = NetworkAnomalyDetectionPipeline()
    
    # Run complete pipeline
    if args.data:
        results = pipeline.run_complete_pipeline(data_path=args.data, save_models=not args.no_save)
    else:
        # Update config for custom parameters
        config = {
            'data_generation': {
                'n_samples': args.samples,
                'anomaly_ratio': args.anomaly_ratio
            }
        }
        pipeline = NetworkAnomalyDetectionPipeline(config)
        results = pipeline.run_complete_pipeline(save_models=not args.no_save)
    
    # Generate detailed report
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION REPORT")
    print("=" * 60)
    
    evaluator = AnomalyEvaluator()
    
    for model_name, metrics in results['evaluation_results'].items():
        print(f"\n{model_name.upper()} PERFORMANCE:")
        print("-" * 40)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Anomaly   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Save results
    results_df = pd.DataFrame(results['evaluation_results']).T
    results_df.to_csv(os.path.join(args.output, 'evaluation_results.csv'))
    print(f"\nResults saved to {args.output}/evaluation_results.csv")
    
    # Demo mode: create visualizations
    if args.demo:
        print("\nGenerating visualizations...")
        
        # Get test data for visualization
        y_test = pipeline.y_test
        X_test = pipeline.X_test
        
        # Create visualizations for the best performing model
        best_model = max(results['evaluation_results'].items(), 
                         key=lambda x: x[1]['roc_auc'])[0]
        
        if best_model == 'ensemble':
            model = pipeline.ensemble
        else:
            model = pipeline.trained_models[best_model]
        
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)
        
        # Create dashboard
        dashboard = evaluator.create_dashboard(
            y_test, y_pred, y_scores, 
            model_name=best_model,
            timestamps=pipeline.processed_data['timestamp'].iloc[-len(y_test):] if 'timestamp' in pipeline.processed_data.columns else None,
            feature_names=pipeline.feature_extractor.get_feature_names(),
            feature_importance=None
        )
        
        # Save visualizations
        for name, fig in dashboard.items():
            fig.write_html(os.path.join(args.output, f'{name}.html'))
        
        print(f"Visualizations saved to {args.output}/")
    
    # Detect anomalies on full dataset
    print("\nDetecting anomalies on full dataset...")
    y_pred, y_scores, annotated_data = pipeline.detect_anomalies()
    
    # Save annotated data
    annotated_data.to_csv(os.path.join(args.output, 'annotated_data.csv'), index=False)
    
    # Print summary
    total_anomalies = y_pred.sum()
    anomaly_rate = y_pred.mean()
    
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {len(y_pred)}")
    print(f"Anomalies detected: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")
    print(f"Best performing model: {best_model}")
    print(f"Results saved to: {args.output}/")
    
    # Show some example anomalies
    if total_anomalies > 0:
        print(f"\nTop 5 detected anomalies:")
        anomaly_data = annotated_data[annotated_data['predicted_anomaly'] == 1].nlargest(5, 'anomaly_score')
        
        for idx, row in anomaly_data.iterrows():
            print(f"  - Score: {row['anomaly_score']:.4f}, "
                  f"Src: {row.get('src_ip', 'N/A')}, "
                  f"Dst: {row.get('dst_ip', 'N/A')}, "
                  f"Port: {row.get('dst_port', 'N/A')}, "
                  f"Protocol: {row.get('protocol', 'N/A')}")


if __name__ == "__main__":
    main()
