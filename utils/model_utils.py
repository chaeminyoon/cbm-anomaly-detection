"""
Model loading and prediction utilities
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
from typing import Tuple, Dict, List, Union
import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

@st.cache_data
def resolve_run_id() -> Union[str, None]:
    """Resolve the model run id: pinned via config, else latest IsolationForest run"""
    if config.MLFLOW_RUN_ID:
        return config.MLFLOW_RUN_ID
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.model_type = 'IsolationForest'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else None
    except Exception:
        return None

@st.cache_resource
def load_model():
    """Load Isolation Forest model from MLflow with caching"""
    run_id = resolve_run_id()
    if run_id is None:
        st.error("No trained IsolationForest run found in MLflow. "
                 "Run training/train_models_mlflow.py first.")
        return None
    try:
        model = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

@st.cache_data
def load_training_data():
    """Load preprocessed training data with caching"""
    try:
        df = pd.read_parquet(config.DATA_PATH)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in feature_cols:
            feature_cols.remove('timestamp')
        return df, feature_cols
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None, None

def prepare_scaler(df, feature_cols):
    """Prepare StandardScaler fitted on training data"""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler

def predict_anomaly(model, scaler, feature_cols, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[int, float]:
    """
    Predict anomaly for new data

    Args:
        model: Trained Isolation Forest model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        data: Input data (DataFrame or numpy array)

    Returns:
        prediction: 1 (normal) or -1 (anomaly)
        score: Anomaly score (lower = more anomalous)
    """
    if isinstance(data, pd.DataFrame):
        data = data[feature_cols].values

    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    score = model.score_samples(data_scaled)[0]

    return prediction, score

def analyze_anomaly_causes(model, scaler, feature_cols, data, df_normal) -> List[Dict]:
    """
    Analyze which features contribute most to anomaly

    Returns:
        List of dicts with feature name, z-score, actual value, normal average
    """
    if isinstance(data, pd.DataFrame):
        data_values = data[feature_cols].values.flatten()
    else:
        data_values = data.flatten()

    data_scaled = scaler.transform(data_values.reshape(1, -1))[0]

    # Calculate normal statistics
    normal_scaled = scaler.transform(df_normal[feature_cols])
    normal_mean = normal_scaled.mean(axis=0)
    normal_std = normal_scaled.std(axis=0)

    # Calculate Z-scores
    z_scores = np.abs((data_scaled - normal_mean) / (normal_std + 1e-8))

    # Get top contributing features
    top_indices = np.argsort(z_scores)[-10:][::-1]

    results = []
    for idx in top_indices:
        feat_name = feature_cols[idx]
        results.append({
            'feature': feat_name,
            'z_score': z_scores[idx],
            'actual_value': data_values[idx],
            'normal_avg': df_normal[feat_name].mean(),
            'deviation_pct': ((data_values[idx] - df_normal[feat_name].mean()) /
                            (df_normal[feat_name].mean() + 1e-8)) * 100
        })

    return results

def get_feature_importance(model, scaler, feature_cols, df, predictions) -> pd.DataFrame:
    """
    Calculate overall feature importance based on anomaly contributions

    Returns:
        DataFrame with features and their importance scores
    """
    anomaly_indices = np.where(predictions == -1)[0]

    if len(anomaly_indices) == 0:
        return pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})

    X = scaler.transform(df[feature_cols])
    anomaly_samples = X[anomaly_indices]
    normal_samples = X[predictions == 1]

    normal_mean = normal_samples.mean(axis=0)
    normal_std = normal_samples.std(axis=0)

    # Average Z-score across all anomalies
    avg_z_scores = np.abs((anomaly_samples - normal_mean) / (normal_std + 1e-8)).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_z_scores
    }).sort_values('importance', ascending=False)

    return importance_df

def calculate_statistics(df, predictions, anomaly_scores) -> Dict:
    """Calculate overall statistics for dashboard"""
    return {
        'total_samples': len(predictions),
        'normal_count': (predictions == 1).sum(),
        'anomaly_count': (predictions == -1).sum(),
        'anomaly_rate': ((predictions == -1).sum() / len(predictions)) * 100,
        'avg_anomaly_score': anomaly_scores.mean(),
        'min_anomaly_score': anomaly_scores.min(),
        'max_anomaly_score': anomaly_scores.max(),
    }
