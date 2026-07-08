"""
Statistics and Trends Page - Historical analysis and monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import (
    load_model, load_training_data, prepare_scaler,
    get_feature_importance
)

st.set_page_config(
    page_title="Statistics & Trends",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Statistics & Trend Analysis")
st.markdown("Monitor system health and detect trends over time")
st.markdown("---")

# Load model and data
@st.cache_resource
def initialize():
    model = load_model()
    df, feature_cols = load_training_data()
    if model is None or df is None:
        return None, None, None

    scaler = prepare_scaler(df, feature_cols)
    return model, scaler, feature_cols

model, scaler, feature_cols = initialize()

if model is None:
    st.error("Failed to load model. Please check configuration.")
    st.stop()

# Load and prepare data with simulated timestamps
@st.cache_data
def prepare_time_series_data():
    df, _ = load_training_data()

    # Simulate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=len(df))

    df['timestamp'] = timestamps
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()

    # Get predictions
    X = scaler.transform(df[feature_cols])
    df['prediction'] = model.predict(X)
    df['anomaly_score'] = model.score_samples(X)
    df['is_anomaly'] = df['prediction'] == -1

    return df

with st.spinner("Loading time series data..."):
    ts_df = prepare_time_series_data()

st.success(f"✅ Loaded {len(ts_df):,} samples over {(ts_df['timestamp'].max() - ts_df['timestamp'].min()).days} days")

# Date range selector
st.markdown("## 📅 Time Range Selection")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=ts_df['timestamp'].min().date(),
        min_value=ts_df['timestamp'].min().date(),
        max_value=ts_df['timestamp'].max().date()
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=ts_df['timestamp'].max().date(),
        min_value=ts_df['timestamp'].min().date(),
        max_value=ts_df['timestamp'].max().date()
    )

# Filter data by date range
filtered_df = ts_df[
    (ts_df['timestamp'].dt.date >= start_date) &
    (ts_df['timestamp'].dt.date <= end_date)
]

st.info(f"📊 Showing {len(filtered_df):,} samples from {start_date} to {end_date}")

st.markdown("---")

# Key Metrics
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_anomalies = filtered_df['is_anomaly'].sum()
    st.metric("Total Anomalies", f"{total_anomalies:,}")

with col2:
    anomaly_rate = (total_anomalies / len(filtered_df)) * 100
    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

with col3:
    avg_score = filtered_df['anomaly_score'].mean()
    st.metric("Avg Anomaly Score", f"{avg_score:.4f}")

with col4:
    daily_avg_anomalies = filtered_df.groupby('date')['is_anomaly'].sum().mean()
    st.metric("Avg Daily Anomalies", f"{daily_avg_anomalies:.1f}")

st.markdown("---")

# Visualizations
st.markdown("## 📈 Trend Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "Time Series",
    "Daily Patterns",
    "Feature Trends",
    "Correlation Analysis"
])

with tab1:
    st.markdown("### Anomaly Detection Over Time")

    # Daily aggregation
    daily_stats = filtered_df.groupby('date').agg({
        'is_anomaly': 'sum',
        'anomaly_score': 'mean',
        'prediction': 'count'
    }).reset_index()
    daily_stats.columns = ['date', 'anomaly_count', 'avg_score', 'total_samples']
    daily_stats['anomaly_rate'] = (daily_stats['anomaly_count'] / daily_stats['total_samples']) * 100

    # Create dual-axis plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Anomaly Count", "Daily Anomaly Rate (%)"),
        vertical_spacing=0.12
    )

    # Anomaly count
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['anomaly_count'],
            mode='lines+markers',
            name='Anomaly Count',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Anomaly rate
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['anomaly_rate'],
            mode='lines+markers',
            name='Anomaly Rate (%)',
            line=dict(color='orange', width=2),
            marker=dict(size=6),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%)", row=2, col=1)

    fig.update_layout(height=600, template='plotly_white', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show trend analysis
    col1, col2 = st.columns(2)

    with col1:
        # Calculate trend
        if len(daily_stats) > 1:
            recent_rate = daily_stats.tail(7)['anomaly_rate'].mean()
            earlier_rate = daily_stats.head(7)['anomaly_rate'].mean()
            trend_change = ((recent_rate - earlier_rate) / (earlier_rate + 1e-8)) * 100

            if trend_change > 10:
                st.warning(f"⚠️ **Increasing Trend:** Anomaly rate increased by {trend_change:.1f}%")
            elif trend_change < -10:
                st.success(f"✅ **Decreasing Trend:** Anomaly rate decreased by {abs(trend_change):.1f}%")
            else:
                st.info(f"📊 **Stable Trend:** Anomaly rate changed by {trend_change:+.1f}%")

    with col2:
        # Peak anomaly day
        peak_day = daily_stats.loc[daily_stats['anomaly_count'].idxmax()]
        st.error(f"""
        **Peak Anomaly Day:**
        - Date: {peak_day['date']}
        - Anomalies: {peak_day['anomaly_count']:.0f}
        - Rate: {peak_day['anomaly_rate']:.2f}%
        """)

with tab2:
    st.markdown("### Daily and Hourly Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Anomalies by day of week
        dow_stats = filtered_df.groupby('day_of_week')['is_anomaly'].agg(['sum', 'count'])
        dow_stats['rate'] = (dow_stats['sum'] / dow_stats['count']) * 100
        dow_stats = dow_stats.reset_index()

        # Order days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats['day_of_week'] = pd.Categorical(dow_stats['day_of_week'], categories=day_order, ordered=True)
        dow_stats = dow_stats.sort_values('day_of_week')

        fig = px.bar(
            dow_stats,
            x='day_of_week',
            y='rate',
            color='rate',
            color_continuous_scale='Reds',
            labels={'day_of_week': 'Day of Week', 'rate': 'Anomaly Rate (%)'},
            title='Anomaly Rate by Day of Week'
        )
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Anomalies by hour
        hour_stats = filtered_df.groupby('hour')['is_anomaly'].agg(['sum', 'count'])
        hour_stats['rate'] = (hour_stats['sum'] / hour_stats['count']) * 100
        hour_stats = hour_stats.reset_index()

        fig = px.line(
            hour_stats,
            x='hour',
            y='rate',
            markers=True,
            labels={'hour': 'Hour of Day', 'rate': 'Anomaly Rate (%)'},
            title='Anomaly Rate by Hour of Day'
        )
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Feature Trends Over Time")

    # Select feature to analyze
    selected_feature = st.selectbox(
        "Select feature to analyze",
        feature_cols,
        index=0
    )

    # Calculate rolling statistics
    feature_stats = filtered_df.groupby('date')[selected_feature].agg(['mean', 'std', 'min', 'max']).reset_index()

    fig = go.Figure()

    # Mean line
    fig.add_trace(go.Scatter(
        x=feature_stats['date'],
        y=feature_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='blue', width=2)
    ))

    # Confidence band (mean ± std)
    fig.add_trace(go.Scatter(
        x=feature_stats['date'],
        y=feature_stats['mean'] + feature_stats['std'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=feature_stats['date'],
        y=feature_stats['mean'] - feature_stats['std'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Trend of {selected_feature}",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Mean", f"{filtered_df[selected_feature].mean():.4f}")

    with col2:
        st.metric("Std Dev", f"{filtered_df[selected_feature].std():.4f}")

    with col3:
        st.metric("Min", f"{filtered_df[selected_feature].min():.4f}")

    with col4:
        st.metric("Max", f"{filtered_df[selected_feature].max():.4f}")

with tab4:
    st.markdown("### Feature Correlation with Anomalies")

    # Calculate correlation between features and anomaly occurrence
    correlations = []
    for feat in feature_cols:
        corr = filtered_df[feat].corr(filtered_df['is_anomaly'].astype(int))
        correlations.append({'feature': feat, 'correlation': abs(corr)})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    # Top correlated features
    top_corr = corr_df.head(20)

    fig = px.bar(
        top_corr.iloc[::-1],
        x='correlation',
        y='feature',
        orientation='h',
        color='correlation',
        color_continuous_scale='Blues',
        labels={'correlation': 'Absolute Correlation', 'feature': 'Feature'},
        title='Top 20 Features Correlated with Anomalies'
    )

    fig.update_layout(height=600, template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Full Correlation Table"):
        st.dataframe(
            corr_df.style.format({'correlation': '{:.4f}'}),
            use_container_width=True
        )

# Statistical Summary
st.markdown("---")
st.markdown("## 📋 Statistical Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Anomaly Statistics")
    summary_stats = {
        "Total Samples": len(filtered_df),
        "Total Anomalies": int(total_anomalies),
        "Anomaly Rate": f"{anomaly_rate:.2f}%",
        "Average Anomaly Score": f"{avg_score:.6f}",
        "Min Anomaly Score": f"{filtered_df['anomaly_score'].min():.6f}",
        "Max Anomaly Score": f"{filtered_df['anomaly_score'].max():.6f}",
        "Days in Range": (end_date - start_date).days + 1,
        "Avg Anomalies/Day": f"{daily_avg_anomalies:.2f}"
    }

    for key, value in summary_stats.items():
        st.metric(key, value)

with col2:
    st.markdown("### Feature Statistics")

    # Get feature importance
    predictions = filtered_df['prediction'].values
    importance_df = get_feature_importance(model, scaler, feature_cols, filtered_df, predictions)

    st.markdown("**Top 5 Most Important Features:**")
    for i, row in importance_df.head(5).iterrows():
        st.markdown(f"{i+1}. **{row['feature']}** - Importance: {row['importance']:.4f}")

# Export options
st.markdown("---")
st.markdown("## 📥 Export Reports")

col1, col2 = st.columns(2)

with col1:
    # Daily statistics export
    csv_daily = daily_stats.to_csv(index=False)
    st.download_button(
        "📊 Download Daily Statistics",
        csv_daily,
        "daily_statistics.csv",
        "text/csv",
        help="Download daily anomaly statistics"
    )

with col2:
    # Feature importance export
    csv_importance = corr_df.to_csv(index=False)
    st.download_button(
        "📈 Download Feature Correlations",
        csv_importance,
        "feature_correlations.csv",
        "text/csv",
        help="Download feature correlation analysis"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> Use trend analysis to predict maintenance needs and optimize operations</p>
</div>
""", unsafe_allow_html=True)
