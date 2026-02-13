import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AnomalyEvaluator:
    """Evaluate and visualize anomaly detection results."""
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            
            # Derived metrics
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            
            # ROC-AUC
            'roc_auc': auc(*roc_curve(y_true, y_scores)[:2]),
            
            # PR-AUC
            'pr_auc': average_precision_score(y_true, y_scores),
            
            # Classification report
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model") -> go.Figure:
        """Plot confusion matrix using Plotly."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Anomaly'],
            y=['Actual Normal', 'Actual Anomaly'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      model_name: str = "Model") -> go.Figure:
        """Plot ROC curve using Plotly."""
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                   model_name: str = "Model") -> go.Figure:
        """Plot Precision-Recall curve using Plotly."""
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline
        baseline = len(y_true[y_true == 1]) / len(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Baseline ({baseline:.3f})',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray, 
                               model_name: str = "Model") -> go.Figure:
        """Plot distribution of anomaly scores."""
        
        df = pd.DataFrame({
            'score': y_scores,
            'label': y_true
        })
        df['label'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})
        
        fig = go.Figure()
        
        for label in ['Normal', 'Anomaly']:
            data = df[df['label'] == label]['score']
            fig.add_trace(go.Histogram(
                x=data,
                name=label,
                opacity=0.7,
                nbinsx=50
            ))
        
        fig.update_layout(
            title=f'Anomaly Score Distribution - {model_name}',
            xaxis_title='Anomaly Score',
            yaxis_title='Frequency',
            width=600,
            height=400,
            barmode='overlay'
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               top_n: int = 20) -> go.Figure:
        """Plot feature importance."""
        
        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = go.Figure(data=go.Bar(
            x=df['importance'],
            y=df['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            width=800,
            height=max(400, len(df) * 25)
        )
        
        return fig
    
    def plot_anomaly_timeline(self, timestamps: pd.DatetimeIndex, 
                             y_true: np.ndarray, y_pred: np.ndarray,
                             y_scores: np.ndarray) -> go.Figure:
        """Plot anomalies over time."""
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'true_label': y_true,
            'predicted_label': y_pred,
            'anomaly_score': y_scores
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('True vs Predicted Labels', 'Anomaly Scores'),
            vertical_spacing=0.1
        )
        
        # True vs Predicted
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['true_label'],
            mode='markers',
            name='True Anomalies',
            marker=dict(color='red', size=6),
            opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['predicted_label'],
            mode='markers',
            name='Predicted Anomalies',
            marker=dict(color='blue', size=4),
            opacity=0.5
        ), row=1, col=1)
        
        # Anomaly scores
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='green', width=1)
        ), row=2, col=1)
        
        # Add threshold line
        threshold = 0.5
        fig.add_trace(go.Scatter(
            x=[df['timestamp'].min(), df['timestamp'].max()],
            y=[threshold, threshold],
            mode='lines',
            name='Threshold',
            line=dict(color='red', width=2, dash='dash')
        ), row=2, col=1)
        
        fig.update_layout(
            title='Anomaly Detection Timeline',
            width=1000,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_scores: np.ndarray, model_name: str = "Model",
                        timestamps: pd.DatetimeIndex = None,
                        feature_names: List[str] = None,
                        feature_importance: np.ndarray = None) -> Dict[str, go.Figure]:
        """Create a comprehensive dashboard of visualizations."""
        
        dashboard = {}
        
        # Core plots
        dashboard['confusion_matrix'] = self.plot_confusion_matrix(y_true, y_pred, model_name)
        dashboard['roc_curve'] = self.plot_roc_curve(y_true, y_scores, model_name)
        dashboard['pr_curve'] = self.plot_precision_recall_curve(y_true, y_scores, model_name)
        dashboard['score_distribution'] = self.plot_score_distribution(y_true, y_scores, model_name)
        
        # Optional plots
        if timestamps is not None:
            dashboard['timeline'] = self.plot_anomaly_timeline(timestamps, y_true, y_pred, y_scores)
            
        if feature_names is not None and feature_importance is not None:
            dashboard['feature_importance'] = self.plot_feature_importance(feature_names, feature_importance)
        
        return dashboard
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Compare multiple models' performance."""
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * 3] * 2
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            values = [results[model][metric] for model in models]
            
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors[i],
                showlegend=False
            ), row=row, col=col)
        
        fig.update_layout(
            title='Model Performance Comparison',
            width=1200,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_scores: np.ndarray, model_name: str = "Model") -> str:
        """Generate a text report of evaluation results."""
        
        metrics = self.calculate_metrics(y_true, y_pred, y_scores)
        
        report = f"""
# Anomaly Detection Evaluation Report - {model_name}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **Specificity**: {metrics['specificity']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **PR-AUC**: {metrics['pr_auc']:.4f}

## Confusion Matrix
```
              Predicted
              Normal  Anomaly
Actual Normal   {metrics['true_negatives']:6d}  {metrics['false_positives']:6d}
       Anomaly   {metrics['false_negatives']:6d}  {metrics['true_positives']:6d}
```

## Classification Report
```
              Precision  Recall  F1-Score  Support
Normal        {metrics['classification_report']['0']['precision']:.3f}    {metrics['classification_report']['0']['recall']:.3f}   {metrics['classification_report']['0']['f1-score']:.3f}   {metrics['classification_report']['0']['support']:.0f}
Anomaly       {metrics['classification_report']['1']['precision']:.3f}    {metrics['classification_report']['1']['recall']:.3f}   {metrics['classification_report']['1']['f1-score']:.3f}   {metrics['classification_report']['1']['support']:.0f}
```
"""
        
        return report
