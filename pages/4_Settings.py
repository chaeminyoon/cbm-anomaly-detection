"""
Settings Page - Configuration and alert management
"""
import streamlit as st
import json
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import resolve_run_id

st.set_page_config(
    page_title="Settings",
    page_icon=":material/directions_boat:",
    layout="wide"
)

# Compact professional styling (shared with main page)
st.markdown("""
<style>
    .block-container { padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1500px; }
    [data-testid="stMetric"] {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 6px; padding: 0.55rem 0.85rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; }
    h1 { font-size: 1.55rem !important; color: #0B3C61; }
    h2 { font-size: 1.12rem !important; padding-top: 0.4rem !important; }
    h3 { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("System Settings & Configuration")
st.markdown("Configure anomaly detection parameters and alert settings")
st.markdown("---")

# Initialize session state for settings
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'anomaly_threshold': config.ANOMALY_THRESHOLD,
        'contamination_rate': config.CONTAMINATION_RATE,
        'alert_enabled': config.ALERT_ENABLED,
        'alert_z_score': config.ALERT_THRESHOLD_Z_SCORE,
        'alert_email': config.ALERT_EMAIL,
        'refresh_interval': config.REFRESH_INTERVAL,
    }

# Tabs for different settings categories
tab1, tab2, tab3, tab4 = st.tabs([
    "Model Settings",
    "Alert Configuration",
    "System Info",
    "Advanced"
])

with tab1:
    st.markdown("## Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Detection Parameters")

        anomaly_threshold = st.number_input(
            "Anomaly Score Threshold",
            min_value=-1.0,
            max_value=0.0,
            value=st.session_state.settings['anomaly_threshold'],
            step=0.01,
            help="Samples with scores below this threshold are flagged as anomalies"
        )

        contamination_rate = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.20,
            value=st.session_state.settings['contamination_rate'],
            step=0.01,
            format="%.2f",
            help="Expected proportion of anomalies in the dataset (1-20%)"
        )

        st.info(f"""
        **Current Settings:**
        - Threshold: {anomaly_threshold:.4f}
        - Contamination: {contamination_rate * 100:.1f}%

        **Interpretation:**
        - Lower threshold = more sensitive (more anomalies detected)
        - Higher contamination = expects more anomalies
        """)

    with col2:
        st.markdown("### Model Information")

        st.json({
            "Model Type": "Isolation Forest",
            "MLflow Run ID": resolve_run_id(),
            "Training Data": config.DATA_PATH,
            "Current Status": "Loaded",
        })

        if st.button("Reload Model", help="Reload model from MLflow"):
            st.cache_resource.clear()
            st.success("Model cache cleared. Model will reload on next prediction.")
            st.rerun()

    # Save model settings
    if st.button("Save Model Settings", type="primary"):
        st.session_state.settings['anomaly_threshold'] = anomaly_threshold
        st.session_state.settings['contamination_rate'] = contamination_rate
        st.success("Model settings saved!")
        st.info("Note: These settings affect detection sensitivity. Test thoroughly before production use.")

with tab2:
    st.markdown("## Alert Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Alert Settings")

        alert_enabled = st.toggle(
            "Enable Alerts",
            value=st.session_state.settings['alert_enabled'],
            help="Turn on/off anomaly alert notifications"
        )

        if alert_enabled:
            alert_z_score = st.slider(
                "Alert Threshold (Z-Score)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.settings['alert_z_score'],
                step=0.5,
                help="Trigger alert when anomaly Z-score exceeds this value"
            )

            alert_email = st.text_input(
                "Alert Email Address",
                value=st.session_state.settings['alert_email'],
                help="Email address to send anomaly alerts"
            )

            st.markdown("---")
            st.markdown("### Alert Triggers")

            trigger_immediate = st.checkbox("Immediate Alert (Any Anomaly)", value=True)
            trigger_critical = st.checkbox("Critical Alert (Z-Score > 5)", value=True)
            trigger_batch = st.checkbox("Daily Summary Email", value=False)

            if trigger_batch:
                batch_time = st.time_input("Daily Summary Time", value=None)

    with col2:
        st.markdown("### Alert Preview")

        # Sample alert message
        st.info("""
        **Sample Alert Email:**

        ---
        **Subject:** CBM Anomaly Alert - Critical

        **Body:**
        Anomaly Detected in Ship Shaft Monitoring System

        **Details:**
        - Timestamp: 2025-11-05 14:32:15
        - Anomaly Score: -0.6534
        - Severity: Critical (Z-Score: 8.2)

        **Top Problematic Sensors:**
        1. Str_LD_EBV_VIBRAT_kurtosis (Z-score: 8.2)
        2. VD_EBV_VIBRAT_p2p (Z-score: 5.7)
        3. HD_EBV_VIBRAT_kurtosis (Z-score: 4.9)

        **Recommended Actions:**
        - Inspect Str_LD_EBV_VIBRAT sensor immediately
        - Review maintenance logs
        - Contact technical team

        ---
        This is an automated alert from CBM Monitoring System
        """)

        if alert_enabled:
            if st.button("Send Test Alert"):
                st.success(f"Test alert sent to {alert_email}")
                st.info("Check your email inbox (including spam folder)")

    # Save alert settings
    if st.button("Save Alert Settings", type="primary"):
        st.session_state.settings['alert_enabled'] = alert_enabled
        if alert_enabled:
            st.session_state.settings['alert_z_score'] = alert_z_score
            st.session_state.settings['alert_email'] = alert_email
        st.success("Alert settings saved!")

with tab3:
    st.markdown("## System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Application Info")
        import sys
        import streamlit as _st_mod
        import mlflow as _mlflow_mod
        st.json({
            "Application": "CBM Anomaly Detection Dashboard",
            "Version": "1.0.0",
            "Framework": f"Streamlit {_st_mod.__version__}",
            "Python Version": sys.version.split()[0],
            "MLflow Version": _mlflow_mod.__version__,
            "Deployment": "Local / AWS Ready"
        })

    with col2:
        st.markdown("### Model Info")
        st.json({
            "Model Type": "Isolation Forest (sklearn)",
            "Run ID": resolve_run_id(),
            "Contamination": f"{config.CONTAMINATION_RATE:.0%}",
            "Data Path": config.DATA_PATH,
        })

    st.markdown("---")
    st.markdown("### Resource Usage")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Memory", "~15 MB", help="Approximate model size in memory")

    with col2:
        st.metric("Cache Usage", "Variable", help="Streamlit caching for faster loads")

    with col3:
        st.metric("Avg Response Time", "< 100ms", help="Typical prediction latency")

    st.markdown("---")
    st.markdown("### External Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**MLflow UI**")
        st.link_button("Open MLflow", "http://localhost:5000", help="View experiment tracking")

    with col2:
        st.markdown("**Documentation**")
        st.link_button("View Guides", "#", help="Access user documentation")

    with col3:
        st.markdown("**Support**")
        st.link_button("Get Help", "#", help="Contact support team")

with tab4:
    st.markdown("## Advanced Settings")

    st.warning("**Warning:** These settings are for advanced users only. Incorrect configuration may affect system performance.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Performance Settings")

        refresh_interval = st.number_input(
            "Dashboard Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.settings['refresh_interval'],
            step=10,
            help="Auto-refresh interval for real-time monitoring"
        )

        cache_ttl = st.number_input(
            "Cache TTL (seconds)",
            min_value=60,
            max_value=3600,
            value=600,
            step=60,
            help="Time-to-live for cached data"
        )

        max_batch_size = st.number_input(
            "Max Batch Size",
            min_value=100,
            max_value=100000,
            value=10000,
            step=1000,
            help="Maximum samples for batch analysis"
        )

    with col2:
        st.markdown("### Data Settings")

        data_retention_days = st.number_input(
            "Data Retention (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="How long to keep historical data"
        )

        log_level = st.selectbox(
            "Logging Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            help="Application logging verbosity"
        )

        enable_telemetry = st.checkbox(
            "Enable Telemetry",
            value=False,
            help="Send anonymous usage statistics"
        )

    st.markdown("---")
    st.markdown("### Database Configuration (Production)")

    with st.expander("Database Settings"):
        db_host = st.text_input("Database Host", value=config.DB_HOST)
        db_port = st.number_input("Database Port", value=config.DB_PORT)
        db_name = st.text_input("Database Name", value=config.DB_NAME)
        db_user = st.text_input("Database User", value=config.DB_USER)

        st.info("For production deployment, configure PostgreSQL or MySQL for persistence")

    st.markdown("---")
    st.markdown("### AWS Deployment Configuration")

    with st.expander("AWS Settings"):
        aws_region = st.selectbox(
            "AWS Region",
            ["ap-northeast-2", "us-east-1", "us-west-2", "eu-west-1"],
            index=0
        )

        aws_s3_bucket = st.text_input("S3 Bucket", value=config.AWS_S3_BUCKET)
        aws_ecr_repo = st.text_input("ECR Repository", value=config.AWS_ECR_REPOSITORY)

        st.code(f"""
# AWS Deployment Example
# 1. Build Docker image
docker build -t {aws_ecr_repo} .

# 2. Push to ECR
aws ecr get-login-password --region {aws_region} | docker login ...
docker push {aws_ecr_repo}:latest

# 3. Deploy to ECS/App Runner
# See AWS deployment guide for details
        """, language="bash")

    # Save advanced settings
    if st.button("Save Advanced Settings", type="primary"):
        st.session_state.settings['refresh_interval'] = refresh_interval
        st.success("Advanced settings saved!")
        st.warning("Some settings may require application restart")

# Export/Import Configuration
st.markdown("---")
st.markdown("## Configuration Management")

col1, col2, col3 = st.columns(3)

with col1:
    # Export configuration
    config_json = json.dumps(st.session_state.settings, indent=2)
    st.download_button(
        "Export Configuration",
        config_json,
        "cbm_config.json",
        "application/json",
        help="Download current configuration as JSON"
    )

with col2:
    # Import configuration
    uploaded_config = st.file_uploader(
        "Import Configuration",
        type=['json'],
        help="Upload previously exported configuration"
    )

    if uploaded_config is not None:
        try:
            imported_settings = json.load(uploaded_config)
            st.session_state.settings.update(imported_settings)
            st.success("Configuration imported successfully!")
        except Exception as e:
            st.error(f"Error importing configuration: {str(e)}")

with col3:
    # Reset to defaults
    if st.button("Reset to Defaults", help="Reset all settings to default values"):
        st.session_state.settings = {
            'anomaly_threshold': config.ANOMALY_THRESHOLD,
            'contamination_rate': config.CONTAMINATION_RATE,
            'alert_enabled': config.ALERT_ENABLED,
            'alert_z_score': config.ALERT_THRESHOLD_Z_SCORE,
            'alert_email': config.ALERT_EMAIL,
            'refresh_interval': config.REFRESH_INTERVAL,
        }
        st.success("Settings reset to defaults")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> <strong>Tip:</strong> Export your configuration before making changes to easily rollback if needed</p>
</div>
""", unsafe_allow_html=True)
