"""
Real-time Anomaly Detection Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import (
    load_model, load_training_data, prepare_scaler,
    predict_anomaly, analyze_anomaly_causes
)

st.set_page_config(
    page_title="Real-time Detection",
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

st.title("Real-time Anomaly Detection")
st.markdown("Input sensor data to detect anomalies in real-time")
st.markdown("---")

# Load model and data
@st.cache_resource
def initialize():
    model = load_model()
    df, feature_cols = load_training_data()
    if model is None or df is None:
        return None, None, None, None

    scaler = prepare_scaler(df, feature_cols)

    # Get normal data for comparison
    X = scaler.transform(df[feature_cols])
    predictions = model.predict(X)
    df_normal = df[predictions == 1]

    return model, scaler, feature_cols, df_normal

model, scaler, feature_cols, df_normal = initialize()

if model is None:
    st.error("Failed to load model. Please check configuration.")
    st.stop()

# Input methods
st.markdown("## Input Method")
input_method = st.radio(
    "Choose input method:",
    ["Manual Input (Top 10 Features)", "Random Sample", "Upload CSV"]
)

if input_method == "Manual Input (Top 10 Features)":
    st.markdown("### Enter Sensor Values (Top 10 Critical Sensors)")
    st.info("For simplicity, enter values for the top 10 most important sensors. Others will use average values.")

    # Top 10 features based on previous analysis
    top_features = feature_cols[:10]

    col1, col2 = st.columns(2)

    input_data = {}
    for i, feat in enumerate(top_features):
        avg_val = df_normal[feat].mean()
        std_val = df_normal[feat].std()

        with col1 if i % 2 == 0 else col2:
            input_data[feat] = st.number_input(
                f"{feat}",
                value=float(avg_val),
                format="%.6f",
                help=f"Normal range: {avg_val - 2*std_val:.4f} ~ {avg_val + 2*std_val:.4f}"
            )

    # Fill remaining features with average values
    full_input = df_normal[feature_cols].mean().to_dict()
    full_input.update(input_data)

    if st.button("Detect Anomaly", type="primary"):
        # Prepare data
        input_array = np.array([full_input[feat] for feat in feature_cols])

        # Predict
        prediction, score = predict_anomaly(model, scaler, feature_cols, input_array)

        st.markdown("---")
        st.markdown("## Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == -1:
                st.error("### ANOMALY DETECTED!")
                st.markdown("**Status:** Abnormal")
            else:
                st.success("### NORMAL")
                st.markdown("**Status:** Normal")

        with col2:
            st.metric("Anomaly Score", f"{score:.6f}")
            st.caption("Lower score = more anomalous")

        with col3:
            confidence = abs(score) / abs(config.ANOMALY_THRESHOLD)
            st.metric("Confidence", f"{min(confidence * 100, 100):.1f}%")

        if prediction == -1:
            st.markdown("---")
            st.markdown("## Root Cause Analysis")

            # Analyze causes
            input_df = pd.DataFrame([full_input])
            causes = analyze_anomaly_causes(model, scaler, feature_cols, input_df, df_normal)

            st.markdown("### Top 5 Problematic Sensors")

            for i, cause in enumerate(causes[:5], 1):
                with st.expander(f"#{i} - {cause['feature']} (Z-score: {cause['z_score']:.2f})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Actual Value", f"{cause['actual_value']:.4f}")

                    with col2:
                        st.metric("Normal Average", f"{cause['normal_avg']:.4f}")

                    with col3:
                        st.metric("Deviation", f"{cause['deviation_pct']:+.1f}%")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cause['z_score'],
                        title={'text': "Z-Score"},
                        gauge={
                            'axis': {'range': [0, 10]},
                            'bar': {'color': "#C4453C" if cause['z_score'] > 3 else "#D97706"},
                            'steps': [
                                {'range': [0, 2], 'color': "lightgreen"},
                                {'range': [2, 3], 'color': "yellow"},
                                {'range': [3, 10], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "#C4453C", 'width': 4},
                                'thickness': 0.75,
                                'value': 3
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### Recommended Actions")
            st.warning(f"""
            1. **Immediate:** Inspect {causes[0]['feature']} sensor
            2. **Short-term:** Check correlation between top 3 sensors
            3. **Medium-term:** Review maintenance logs for patterns
            4. **Alert:** Notify maintenance team immediately
            """)

elif input_method == "Random Sample":
    st.markdown("### Test with Random Sample")

    col1, col2 = st.columns([1, 3])

    with col1:
        sample_type = st.selectbox(
            "Sample Type",
            ["Normal Sample", "Random Anomaly", "Known Anomaly"]
        )

    with col2:
        if st.button("Generate Random Sample", type="primary"):
            st.session_state['random_sample'] = True

    if st.session_state.get('random_sample', False):
        # Get sample based on type
        if sample_type == "Normal Sample":
            sample = df_normal.sample(1)
        else:
            # Get anomaly samples
            X = scaler.transform(df_normal[feature_cols])
            predictions = model.predict(X)
            anomalies = df_normal[predictions == -1]

            if len(anomalies) > 0:
                sample = anomalies.sample(1)
            else:
                sample = df_normal.sample(1)

        # Predict
        prediction, score = predict_anomaly(model, scaler, feature_cols, sample)

        col1, col2 = st.columns(2)

        with col1:
            if prediction == -1:
                st.error("### ANOMALY DETECTED!")
            else:
                st.success("### NORMAL")

        with col2:
            st.metric("Anomaly Score", f"{score:.6f}")

        # Show sample data
        st.markdown("### Sample Data")
        st.dataframe(sample[feature_cols].T, use_container_width=True)

else:  # Upload CSV
    st.markdown("### Upload CSV File")

    uploaded_file = st.file_uploader(
        "Upload CSV file with sensor data",
        type=['csv'],
        help="CSV must contain the same features as training data"
    )

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)

            st.success(f"File uploaded: {upload_df.shape[0]} samples")

            # Check if features match
            missing_features = set(feature_cols) - set(upload_df.columns)

            if missing_features:
                st.error(f"Missing features: {missing_features}")
            else:
                if st.button("Detect Anomalies", type="primary"):
                    # Predict
                    predictions = []
                    scores = []

                    for idx in range(len(upload_df)):
                        sample = upload_df.iloc[idx]
                        pred, score = predict_anomaly(model, scaler, feature_cols, sample)
                        predictions.append(pred)
                        scores.append(score)

                    upload_df['prediction'] = predictions
                    upload_df['anomaly_score'] = scores

                    # Display results
                    st.markdown("### Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Samples", len(upload_df))

                    with col2:
                        anomaly_count = (upload_df['prediction'] == -1).sum()
                        st.metric("Anomalies", anomaly_count)

                    with col3:
                        anomaly_rate = (anomaly_count / len(upload_df)) * 100
                        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

                    # Show anomalies
                    if anomaly_count > 0:
                        st.markdown("### Detected Anomalies")
                        anomaly_df = upload_df[upload_df['prediction'] == -1]
                        st.dataframe(anomaly_df, use_container_width=True)

                        # Download button
                        csv = anomaly_df.to_csv(index=False)
                        st.download_button(
                            "Download Anomalies CSV",
                            csv,
                            "anomalies.csv",
                            "text/csv"
                        )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.info("**Tip:** For production use, integrate this with real-time data streams via API or message queues.")
