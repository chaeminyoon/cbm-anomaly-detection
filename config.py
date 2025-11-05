"""
CBM Anomaly Detection Dashboard - Configuration
"""
import os

# MLflow Configuration
MLFLOW_RUN_ID = "c66eb2b2ae5e4057b839264080ce24e5"  # Isolation Forest Run ID
MLFLOW_TRACKING_URI = "./mlruns"

# Data Configuration
DATA_PATH = "preprocessed_shaft_data.parquet"
FEATURE_COLUMNS = [
    'Str_LD_EBV_VIBRAT_kurtosis', 'VD_EBV_VIBRAT_kurtosis',
    'HD_EBV_VIBRAT_kurtosis', 'Str_LD_EBV_VIBRAT_p2p',
    'VD_EBV_VIBRAT_p2p', 'Str_LD_EBV_VIBRAT_skewness',
    'Str_TD_EBV_VIBRAT_p2p', 'HD_EBV_VIBRAT_p2p',
    'HD_EBV_VIBRAT_skewness', 'Str_LD_EBV_VIBRAT_mean',
    # ... 더 많은 특징들
]

# Model Configuration
ANOMALY_THRESHOLD = -0.50  # Anomaly score threshold from visualization
CONTAMINATION_RATE = 0.05  # 5% expected anomaly rate

# Alert Configuration
ALERT_ENABLED = True
ALERT_THRESHOLD_Z_SCORE = 3.0  # Z-score > 3.0 triggers alert
ALERT_EMAIL = "maintenance@example.com"

# Dashboard Configuration
PAGE_TITLE = "Ship Shaft CBM Monitoring"
PAGE_ICON = "🚢"
LAYOUT = "wide"

# AWS Deployment Configuration (for production)
AWS_REGION = "ap-northeast-2"  # Seoul
AWS_S3_BUCKET = "cbm-anomaly-models"
AWS_ECR_REPOSITORY = "cbm-dashboard"

# Database Configuration (for production)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "cbm_monitoring"
DB_USER = "cbm_user"

# Monitoring Configuration
REFRESH_INTERVAL = 60  # seconds
MAX_HISTORY_DAYS = 30
BATCH_SIZE = 1000
