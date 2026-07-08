"""
Isolation Forest를 사용한 이상 데이터 원인 분석 데모

이 스크립트는 학습된 Isolation Forest 모델로:
1. 이상 데이터 감지
2. 어떤 센서/특징이 이상을 일으켰는지 분석
3. 구체적인 원인 파악
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Windows 환경에서 UTF-8 인코딩 설정
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("🔍 Isolation Forest 이상 데이터 원인 분석 시스템")
print("="*80)

# 1. MLflow에서 학습된 Isolation Forest 모델 로드 (최신 런 자동 탐색)
print("\n📦 Step 1: MLflow에서 모델 로드 중...")
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = MlflowClient()
experiment = client.get_experiment_by_name('CBM_Shaft_Anomaly_Detection')
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id] if experiment else [],
    filter_string="params.model_type = 'IsolationForest'",
    order_by=["start_time DESC"], max_results=1,
) if experiment else []

if not runs:
    print("❌ 학습된 Isolation Forest 런이 없습니다. training/train_models_mlflow.py 를 먼저 실행하세요.")
    exit(1)

run_id = runs[0].info.run_id
model = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
print(f"✅ 모델 로드 완료! (Run ID: {run_id})")

# 2. 데이터 로드 및 전처리
print("\n📂 Step 2: 데이터 로드 중...")
df = pd.read_parquet('preprocessed_shaft_data.parquet')
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'timestamp' in feature_cols:
    feature_cols.remove('timestamp')

print(f"   데이터 shape: {df.shape}")
print(f"   특징 개수: {len(feature_cols)}")
print(f"   특징 목록: {feature_cols[:5]}... (총 {len(feature_cols)}개)")

# 정규화 (학습 시와 동일한 방법)
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])

# 3. 전체 데이터에 대해 이상 예측
print("\n🔍 Step 3: 이상 데이터 탐지 중...")
predictions = model.predict(X)
anomaly_scores = model.score_samples(X)  # 이상 점수 (낮을수록 이상)

n_anomalies = (predictions == -1).sum()
anomaly_rate = (n_anomalies / len(predictions)) * 100

print(f"   전체 데이터: {len(predictions):,}개")
print(f"   이상 데이터: {n_anomalies:,}개 ({anomaly_rate:.2f}%)")
print(f"   정상 데이터: {(predictions == 1).sum():,}개")

# 4. 이상 데이터 샘플 선택 및 분석
print("\n🎯 Step 4: 이상 데이터 상위 5개 분석")
print("-"*80)

# 이상 점수가 가장 낮은 (가장 이상한) 5개 샘플 선택
anomaly_indices = np.where(predictions == -1)[0]
top_anomalies_idx = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])[:5]]

for i, idx in enumerate(top_anomalies_idx, 1):
    print(f"\n【이상 데이터 #{i}】")
    print(f"   인덱스: {idx}")
    print(f"   이상 점수: {anomaly_scores[idx]:.6f} (낮을수록 이상)")

    # 해당 샘플의 원본 값
    sample = df.iloc[idx][feature_cols]
    sample_scaled = X[idx]

    # 각 특징의 기여도 분석
    # 방법: 정상 데이터의 평균/표준편차와 비교
    normal_indices = np.where(predictions == 1)[0]
    normal_mean = X[normal_indices].mean(axis=0)
    normal_std = X[normal_indices].std(axis=0)

    # Z-score 계산 (정규화된 값에서)
    z_scores = np.abs((sample_scaled - normal_mean) / (normal_std + 1e-8))

    # 가장 이상한 상위 5개 특징
    top_features_idx = np.argsort(z_scores)[-5:][::-1]

    print(f"\n   🚨 이상 원인 (상위 5개 센서):")
    for rank, feat_idx in enumerate(top_features_idx, 1):
        feat_name = feature_cols[feat_idx]
        z_score = z_scores[feat_idx]
        actual_value = sample[feat_name]
        normal_avg = df.iloc[normal_indices][feat_name].mean()

        deviation = ((actual_value - normal_avg) / normal_avg * 100) if normal_avg != 0 else 0

        print(f"      {rank}. {feat_name:30s}: Z-score={z_score:.2f}")
        print(f"         실제값: {actual_value:10.4f} | 정상평균: {normal_avg:10.4f} | 편차: {deviation:+.1f}%")

# 5. Feature Importance 분석 (전체 모델 기준)
print("\n" + "="*80)
print("📊 Step 5: 전체 모델의 Feature Importance 분석")
print("-"*80)

# Isolation Forest에서 직접 feature importance를 제공하지 않으므로,
# 이상 데이터에서 각 특징의 평균 기여도를 계산
if len(anomaly_indices) > 0:
    # 이상 데이터의 평균 Z-score
    anomaly_samples = X[anomaly_indices]
    normal_mean = X[predictions == 1].mean(axis=0)
    normal_std = X[predictions == 1].std(axis=0)

    avg_z_scores = np.abs((anomaly_samples - normal_mean) / (normal_std + 1e-8)).mean(axis=0)

    # 상위 10개 중요 특징
    top_10_features = np.argsort(avg_z_scores)[-10:][::-1]

    print("\n🔝 이상 탐지에 가장 중요한 센서 Top 10:")
    for rank, feat_idx in enumerate(top_10_features, 1):
        feat_name = feature_cols[feat_idx]
        importance = avg_z_scores[feat_idx]
        print(f"   {rank:2d}. {feat_name:35s}: {importance:.4f}")

# 6. 시각화
print("\n📈 Step 6: 시각화 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Isolation Forest 이상 탐지 상세 분석', fontsize=16, fontweight='bold')

# 6-1. 이상 점수 분포
ax1 = axes[0, 0]
ax1.hist(anomaly_scores[predictions == 1], bins=50, alpha=0.7, label='Normal', color='green')
ax1.hist(anomaly_scores[predictions == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
ax1.set_xlabel('Anomaly Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6-2. Feature Importance (Top 10)
ax2 = axes[0, 1]
top_10_names = [feature_cols[i] for i in top_10_features]
top_10_values = [avg_z_scores[i] for i in top_10_features]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
ax2.barh(range(10), top_10_values[::-1], color=colors[::-1])
ax2.set_yticks(range(10))
ax2.set_yticklabels(top_10_names[::-1], fontsize=10)
ax2.set_xlabel('Average Z-Score (Anomaly Contribution)', fontsize=12)
ax2.set_title('Top 10 Important Features', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 6-3. 정상 vs 이상 데이터 비율
ax3 = axes[1, 0]
labels = ['Normal', 'Anomaly']
sizes = [(predictions == 1).sum(), (predictions == -1).sum()]
colors_pie = ['#90EE90', '#FF6B6B']
explode = (0, 0.1)
ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax3.set_title('Normal vs Anomaly Distribution', fontsize=14, fontweight='bold')

# 6-4. 상위 3개 이상 샘플의 Feature Profile
ax4 = axes[1, 1]
for i, idx in enumerate(top_anomalies_idx[:3]):
    sample = df.iloc[idx][feature_cols]
    sample_scaled = X[idx]
    z_scores_sample = np.abs((sample_scaled - normal_mean) / (normal_std + 1e-8))
    top_5_for_sample = np.argsort(z_scores_sample)[-5:][::-1]

    feat_names = [feature_cols[j][:15] for j in top_5_for_sample]
    feat_values = [z_scores_sample[j] for j in top_5_for_sample]

    x_pos = np.arange(5)
    ax4.plot(x_pos, feat_values, marker='o', linewidth=2, markersize=8, label=f'Anomaly #{i+1}')

ax4.set_xticks(range(5))
ax4.set_xticklabels(['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'], fontsize=10)
ax4.set_ylabel('Z-Score', fontsize=12)
ax4.set_title('Top 3 Anomalies - Feature Contribution Profile', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isolation_forest_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 시각화 저장 완료: isolation_forest_detailed_analysis.png")

# 7. 실무 활용 예제
print("\n" + "="*80)
print("💼 Step 7: 실무 활용 예제 - 새로운 데이터 분석")
print("-"*80)

# 새로운 데이터가 들어왔다고 가정
print("\n시나리오: 실시간 모니터링 중 새로운 센서 데이터가 들어옴")

# 임의의 샘플 선택 (실제로는 실시간 데이터)
new_data_idx = np.random.choice(len(df), 1)[0]
new_data = df.iloc[new_data_idx][feature_cols].values.reshape(1, -1)
new_data_scaled = scaler.transform(new_data)

# 예측
prediction = model.predict(new_data_scaled)[0]
score = model.score_samples(new_data_scaled)[0]

print(f"\n입력 데이터 인덱스: {new_data_idx}")
print(f"예측 결과: {'🚨 이상!' if prediction == -1 else '✅ 정상'}")
print(f"이상 점수: {score:.6f}")

if prediction == -1:
    print(f"\n⚠️ 이상이 감지되었습니다!")

    # 어떤 센서가 문제인지 분석
    sample_scaled = new_data_scaled[0]
    z_scores = np.abs((sample_scaled - normal_mean) / (normal_std + 1e-8))
    top_problems = np.argsort(z_scores)[-5:][::-1]

    print(f"\n🔧 점검이 필요한 센서 (우선순위 순):")
    for rank, feat_idx in enumerate(top_problems, 1):
        feat_name = feature_cols[feat_idx]
        z_score = z_scores[feat_idx]
        actual = new_data[0, feat_idx]
        normal_avg = df.iloc[normal_indices][feat_name].mean()

        print(f"   {rank}. {feat_name:30s}")
        print(f"      → 현재값: {actual:10.4f} | 정상평균: {normal_avg:10.4f}")
        print(f"      → Z-score: {z_score:.2f} (이상도)")

    print(f"\n💡 조치 권고:")
    print(f"   1. {feature_cols[top_problems[0]]} 센서 우선 점검")
    print(f"   2. 상위 3개 센서의 상관관계 분석")
    print(f"   3. 정비팀에 알림 발송")
else:
    print(f"\n✅ 정상 데이터입니다. 계속 모니터링...")

# 8. API 활용 예제 코드
print("\n" + "="*80)
print("🔌 Step 8: API 서버 구축 예제 코드")
print("-"*80)

api_example = '''
# FastAPI를 사용한 실시간 이상 탐지 API 예제

from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
import pandas as pd

app = FastAPI()

# 모델 로드 (서버 시작 시 1회)
model = mlflow.sklearn.load_model('runs:/c66eb2b2ae5e4057b839264080ce24e5/model')
scaler = StandardScaler()  # 사전 학습된 scaler 로드 필요

@app.post("/predict")
async def predict_anomaly(data: dict):
    """
    입력: {"sensor_values": [v1, v2, v3, ..., v39]}
    출력: {"is_anomaly": true/false, "score": -0.123, "top_features": [...]}
    """
    # 데이터 전처리
    sensor_values = np.array(data["sensor_values"]).reshape(1, -1)
    sensor_scaled = scaler.transform(sensor_values)

    # 예측
    prediction = model.predict(sensor_scaled)[0]
    score = model.score_samples(sensor_scaled)[0]

    # 원인 분석
    if prediction == -1:
        z_scores = calculate_feature_importance(sensor_scaled)
        top_features = get_top_features(z_scores, k=5)
    else:
        top_features = []

    return {
        "is_anomaly": bool(prediction == -1),
        "anomaly_score": float(score),
        "top_problematic_features": top_features,
        "recommended_action": "Inspect sensors" if prediction == -1 else "Continue monitoring"
    }

# 실행: uvicorn api:app --reload
'''

print(api_example)

print("\n" + "="*80)
print("✅ 분석 완료!")
print("="*80)
print("\n📁 생성된 파일:")
print("   - isolation_forest_detailed_analysis.png")
print("\n🌐 MLflow UI에서 더 많은 정보 확인:")
print("   http://localhost:5000")
print("\n💡 핵심 요약:")
print("   1. Isolation Forest는 이상을 감지하고")
print("   2. 어떤 센서가 문제인지 정확히 알려줍니다")
print("   3. Z-score로 각 센서의 기여도를 정량화합니다")
print("   4. 실시간 API로 즉시 활용 가능합니다")
print("="*80)
