"""
Batch Analysis Page - Upload and analyze multiple samples
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import (
    load_model, load_training_data, prepare_scaler,
    get_feature_importance, calculate_statistics, analyze_anomaly_causes
)

st.set_page_config(
    page_title="Batch Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Batch Data Analysis")
st.markdown("Upload and analyze large datasets for anomaly detection")
st.markdown("---")

# Load model and data
@st.cache_resource
def initialize():
    model = load_model()
    df, feature_cols = load_training_data()
    if model is None or df is None:
        return None, None, None, None

    scaler = prepare_scaler(df, feature_cols)

    # Get normal data
    X = scaler.transform(df[feature_cols])
    predictions = model.predict(X)
    df_normal = df[predictions == 1]

    return model, scaler, feature_cols, df_normal

model, scaler, feature_cols, df_normal = initialize()

if model is None:
    st.error("Failed to load model. Please check configuration.")
    st.stop()

# File upload section
st.markdown("## 📤 Upload Data File")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload CSV or Parquet file",
        type=['csv', 'parquet'],
        help="File must contain the same features as training data"
    )

with col2:
    st.info("""
    **Expected Features:** {0}

    **Supported Formats:**
    - CSV (.csv)
    - Parquet (.parquet)
    """.format(len(feature_cols)))

# Use training data option
use_training_data = st.checkbox("📚 Use training data for demo", value=False)

# Process data
if uploaded_file is not None or use_training_data:

    with st.spinner("Loading data..."):
        if use_training_data:
            batch_df = df_normal.copy()
            st.success(f"✅ Loaded training data: {batch_df.shape[0]:,} samples")
        else:
            try:
                if uploaded_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_parquet(uploaded_file)

                st.success(f"✅ File uploaded: {batch_df.shape[0]:,} samples, {batch_df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.stop()

    # Check features
    missing_features = set(feature_cols) - set(batch_df.columns)
    if missing_features:
        st.error(f"❌ Missing required features: {list(missing_features)[:10]}...")
        st.stop()

    st.markdown("---")
    st.markdown("## ⚙️ Analysis Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        sample_size = st.slider(
            "Sample size (for large datasets)",
            min_value=100,
            max_value=min(50000, len(batch_df)),
            value=min(10000, len(batch_df)),
            step=100,
            help="Analyze a subset for faster processing"
        )

    with col2:
        show_details = st.checkbox("Show detailed anomaly analysis", value=True)

    with col3:
        export_results = st.checkbox("Enable result export", value=True)

    # Analyze button
    if st.button("🔍 Run Batch Analysis", type="primary"):

        with st.spinner("Analyzing data..."):
            # Sample data if needed
            if len(batch_df) > sample_size:
                analysis_df = batch_df.sample(n=sample_size, random_state=42)
                st.info(f"Analyzing {sample_size:,} random samples out of {len(batch_df):,}")
            else:
                analysis_df = batch_df.copy()

            # Predict
            X_batch = scaler.transform(analysis_df[feature_cols])
            predictions = model.predict(X_batch)
            anomaly_scores = model.score_samples(X_batch)

            # Add results to dataframe
            analysis_df['prediction'] = predictions
            analysis_df['anomaly_score'] = anomaly_scores
            analysis_df['is_anomaly'] = predictions == -1

            # Calculate statistics
            stats = calculate_statistics(analysis_df, predictions, anomaly_scores)

        st.success("✅ Analysis complete!")

        st.markdown("---")
        st.markdown("## 📈 Analysis Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyzed", f"{stats['total_samples']:,}")

        with col2:
            st.metric("Normal Samples", f"{stats['normal_count']:,}", f"{100 - stats['anomaly_rate']:.1f}%")

        with col3:
            st.metric(
                "Anomalies Detected",
                f"{stats['anomaly_count']:,}",
                f"{stats['anomaly_rate']:.2f}%",
                delta_color="inverse"
            )

        with col4:
            st.metric("Avg Anomaly Score", f"{stats['avg_anomaly_score']:.4f}")

        # Visualizations
        st.markdown("---")
        st.markdown("## 📊 Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Score Distribution",
            "Feature Importance",
            "Anomaly Timeline",
            "Data Table"
        ])

        with tab1:
            # Score distribution
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Anomaly Score Distribution", "Box Plot by Class")
            )

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores[predictions == 1],
                    name='Normal',
                    marker_color='green',
                    opacity=0.7,
                    nbinsx=50
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores[predictions == -1],
                    name='Anomaly',
                    marker_color='red',
                    opacity=0.7,
                    nbinsx=50
                ),
                row=1, col=1
            )

            # Box plot
            fig.add_trace(
                go.Box(
                    y=anomaly_scores[predictions == 1],
                    name='Normal',
                    marker_color='green'
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Box(
                    y=anomaly_scores[predictions == -1],
                    name='Anomaly',
                    marker_color='red'
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Anomaly Score", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Anomaly Score", row=1, col=2)

            fig.update_layout(height=400, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Feature importance
            importance_df = get_feature_importance(model, scaler, feature_cols, analysis_df, predictions)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Bar chart
                top_20 = importance_df.head(20)
                fig = px.bar(
                    top_20.iloc[::-1],
                    x='importance',
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Reds',
                    labels={'importance': 'Avg Z-Score', 'feature': 'Feature'}
                )
                fig.update_layout(height=600, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📋 Top 10 Features")
                st.dataframe(
                    importance_df.head(10).style.format({'importance': '{:.4f}'}),
                    use_container_width=True,
                    height=600
                )

        with tab3:
            # Timeline (simulated with indices)
            st.markdown("### Anomaly Detection Timeline")

            # Create timeline data
            timeline_df = analysis_df[['prediction', 'anomaly_score']].copy()
            timeline_df['index'] = range(len(timeline_df))
            timeline_df['color'] = timeline_df['prediction'].map({1: 'Normal', -1: 'Anomaly'})

            fig = px.scatter(
                timeline_df,
                x='index',
                y='anomaly_score',
                color='color',
                color_discrete_map={'Normal': 'green', 'Anomaly': 'red'},
                labels={'index': 'Sample Index', 'anomaly_score': 'Anomaly Score'},
                hover_data=['prediction']
            )

            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Data table
            st.markdown("### 📋 Detailed Results")

            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                filter_type = st.selectbox(
                    "Filter results",
                    ["All Data", "Anomalies Only", "Normal Only"]
                )

            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Anomaly Score (Low to High)", "Anomaly Score (High to Low)", "Original Order"]
                )

            # Apply filters
            display_df = analysis_df.copy()

            if filter_type == "Anomalies Only":
                display_df = display_df[display_df['is_anomaly']]
            elif filter_type == "Normal Only":
                display_df = display_df[~display_df['is_anomaly']]

            if sort_by == "Anomaly Score (Low to High)":
                display_df = display_df.sort_values('anomaly_score')
            elif sort_by == "Anomaly Score (High to Low)":
                display_df = display_df.sort_values('anomaly_score', ascending=False)

            # Show data
            st.dataframe(
                display_df[['prediction', 'anomaly_score', 'is_anomaly'] + feature_cols[:5]],
                use_container_width=True,
                height=400
            )

        # Detailed anomaly analysis
        if show_details and stats['anomaly_count'] > 0:
            st.markdown("---")
            st.markdown("## 🔍 Detailed Anomaly Analysis")

            # Get worst anomalies
            anomaly_df = analysis_df[analysis_df['is_anomaly']].sort_values('anomaly_score')
            top_anomalies = anomaly_df.head(min(5, len(anomaly_df)))

            st.markdown(f"### Top {len(top_anomalies)} Most Severe Anomalies")

            for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
                with st.expander(f"Anomaly #{idx} - Score: {row['anomaly_score']:.6f}"):

                    # Analyze causes
                    sample_df = pd.DataFrame([row[feature_cols]])
                    causes = analyze_anomaly_causes(model, scaler, feature_cols, sample_df, df_normal)

                    st.markdown("#### Top 5 Contributing Factors")

                    for i, cause in enumerate(causes[:5], 1):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"**{i}. {cause['feature'][:20]}...**")

                        with col2:
                            st.metric("Z-Score", f"{cause['z_score']:.2f}")

                        with col3:
                            st.metric("Actual", f"{cause['actual_value']:.4f}")

                        with col4:
                            st.metric("Deviation", f"{cause['deviation_pct']:+.1f}%")

        # Export results
        if export_results:
            st.markdown("---")
            st.markdown("## 📥 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Export all results
                csv_all = analysis_df.to_csv(index=False)
                st.download_button(
                    "📄 Download All Results (CSV)",
                    csv_all,
                    "batch_analysis_full.csv",
                    "text/csv",
                    help="Download complete analysis results"
                )

            with col2:
                # Export anomalies only
                if stats['anomaly_count'] > 0:
                    csv_anomalies = analysis_df[analysis_df['is_anomaly']].to_csv(index=False)
                    st.download_button(
                        "⚠️ Download Anomalies Only (CSV)",
                        csv_anomalies,
                        "batch_analysis_anomalies.csv",
                        "text/csv",
                        help="Download only detected anomalies"
                    )

            with col3:
                # Export feature importance
                csv_importance = importance_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Feature Importance (CSV)",
                    csv_importance,
                    "feature_importance.csv",
                    "text/csv",
                    help="Download feature importance scores"
                )

else:
    # Instructions
    st.info("""
    ### 📋 Instructions

    1. **Upload a data file** (CSV or Parquet format)
       - File must contain the same {0} features as training data
       - Supports large datasets (up to 50,000 samples for analysis)

    2. **Or use training data** by checking the demo checkbox

    3. **Configure analysis settings**
       - Adjust sample size for faster processing
       - Enable/disable detailed analysis
       - Choose export options

    4. **Run analysis** and view comprehensive results

    ### 📦 File Format Example

    Your CSV/Parquet should have columns like:
    ```
    {1}, {2}, {3}, ...
    ```

    ### 💡 Use Cases

    - **Daily Monitoring:** Upload yesterday's sensor logs
    - **Historical Analysis:** Analyze patterns over weeks/months
    - **Batch Validation:** Test new sensor configurations
    - **Report Generation:** Export results for management
    """.format(len(feature_cols), feature_cols[0], feature_cols[1], feature_cols[2]))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> For production batch processing, consider using Apache Airflow or AWS Batch for automation</p>
</div>
""", unsafe_allow_html=True)
