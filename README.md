# 🚢 Ship Shaft CBM Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.0-0194E2?logo=mlflow)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-ready anomaly detection system for ship propulsion shaft condition-based maintenance (CBM) using Isolation Forest and MLflow.**

## 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [시스템 구조](#시스템-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [모델 설명](#모델-설명)
- [파일 구조](#파일-구조)

---

## 시스템 개요

본 시스템은 선박 추진축계의 진동, RPM 등 센서 데이터를 실시간으로 분석하여:
1. **이상 탐지**: 비정상 동작 패턴 자동 감지
2. **예지 보전**: 고장 전 사전 경고 및 잔여 수명 예측
3. **실시간 모니터링**: 인터랙티브 대시보드를 통한 상태 감시

### 데이터 구조
- **VSL_NO**: 선박번호
- **TRNNG_SHP_NAME**: 선박명 (한바다)
- **MSRM_DTM**: 측정일시
- **RTTN_SPDMTR**: 회전속도계 (RPM)
- **SH_LD_EBY_VIBRAT**: 축계 하단 진동
- **SH_TD_EBY_VIBRAT**: 축계 상단 진동
- **VD_EBY_VIBRAT**: 수직 진동
- **HD_EBY_VIBRAT**: 수평 진동

---

## 주요 기능

### 1. 데이터 전처리
- 대용량 JSON 스트리밍 처리
- 결측치/이상치 처리
- 특징 엔지니어링 (RMS, Peak-to-Peak, Kurtosis, Skewness, FFT 등)
- 데이터 정규화

### 2. 이상 탐지
- **Isolation Forest**: 비지도 학습 기반 이상치 탐지
- **Autoencoder**: 딥러닝 기반 재구성 오차 분석
- **LSTM Autoencoder**: 시계열 패턴 기반 이상 탐지

### 3. 예지 보전
- **건강 지수(Health Index)**: 0~100 스케일의 상태 점수
- **RUL 예측**: Random Forest, Gradient Boosting, LSTM 기반
- 고장 확률 추정

### 4. 실시간 모니터링 대시보드
- Streamlit 기반 인터랙티브 UI
- 실시간 센서 데이터 시각화
- 이상 탐지 알림
- 유지보수 권장사항

---

## 시스템 구조

```
┌─────────────────┐
│  Raw JSON Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Preprocessing     │
│  - Streaming Load       │
│  - Feature Engineering  │
│  - Normalization        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Anomaly Detection      │
│  - Isolation Forest     │
│  - Autoencoder          │
│  - LSTM Autoencoder     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Predictive Maintenance │
│  - Health Index         │
│  - RUL Prediction       │
│  - Failure Prediction   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Real-time Dashboard    │
│  - Streamlit UI         │
│  - Live Monitoring      │
│  - Alerts & Warnings    │
└─────────────────────────┘
```

---

## 설치 방법

### 1. Python 환경 설정
```bash
# Python 3.8 이상 필요
python --version

# 가상환경 생성 (권장)
python -m venv cbm_env

# 가상환경 활성화
# Windows:
cbm_env\Scripts\activate
# Mac/Linux:
source cbm_env/bin/activate
```

### 2. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 데이터 준비
- `한바다호+선박의+추진축계+모니터링+데이터.json` 파일을 현재 디렉토리에 배치
- `HBD_추진축계_데이터정의서.xlsx` 파일 확인

---

## 사용 방법

### 전체 프로세스 순서

#### Step 1: 데이터 전처리
```bash
python cbm_preprocessing.py
```
**출력 파일:**
- `preprocessed_shaft_data.parquet` - 전처리된 데이터

**처리 내용:**
- JSON 스트리밍 로드 (100만 레코드 샘플)
- 결측치 보간
- 이상치 탐지 및 마킹
- 특징 엔지니어링 (40+ 특징 생성)
- 데이터 정규화

#### Step 2: 이상 탐지 모델 학습
```bash
python cbm_anomaly_detection.py
```
**출력 파일:**
- `isolation_forest_model.pkl`
- `autoencoder_model.h5` + `autoencoder_model_threshold.npy`
- `lstm_autoencoder_model.h5` + `lstm_autoencoder_model_threshold.npy`
- 시각화 이미지들 (*.png)

**학습 모델:**
1. Isolation Forest - 빠른 이상치 탐지
2. Autoencoder - 정밀한 패턴 분석
3. LSTM Autoencoder - 시계열 패턴 학습

#### Step 3: 예지 보전 모델 학습
```bash
python cbm_predictive_maintenance.py
```
**출력 파일:**
- `random_forest_rul_model.pkl`
- `gradient_boosting_rul_model.pkl`
- `lstm_rul_model.h5`
- `health_index_timeline.png`
- `rul_predictions_comparison.csv`

**예측 내용:**
- 건강 지수 계산 (0-100)
- RUL (Remaining Useful Life) 예측
- 고장 예측

#### Step 4: 실시간 대시보드 실행
```bash
streamlit run cbm_dashboard.py
```

**대시보드 접속:**
- 브라우저에서 자동으로 열림 (보통 http://localhost:8501)

---

## 모델 설명

### 1. Isolation Forest
- **알고리즘**: 앙상블 기반 이상치 탐지
- **장점**: 빠른 학습 및 추론, 고차원 데이터에 강함
- **용도**: 실시간 이상 탐지, 1차 필터링
- **contamination**: 0.05 (5% 이상치 가정)

### 2. Autoencoder
- **구조**: Encoder (64→32→32) → Latent (32) → Decoder (32→32→64)
- **장점**: 정상 패턴 학습, 미세한 이상 탐지
- **임계값**: 재구성 오차 95 percentile
- **활성화 함수**: ReLU + Dropout(0.2)

### 3. LSTM Autoencoder
- **구조**: LSTM(128→64) → Latent(32) → LSTM(64→128)
- **장점**: 시계열 의존성 학습, 동적 패턴 분석
- **시퀀스 길이**: 100 time steps
- **용도**: 추세 기반 이상 탐지

### 4. RUL 예측 모델
#### Random Forest
- **트리 개수**: 100
- **최대 깊이**: 20
- **용도**: 빠른 예측, Feature Importance 분석

#### Gradient Boosting
- **트리 개수**: 100
- **학습률**: 0.1
- **용도**: 정밀한 RUL 예측

#### LSTM
- **구조**: LSTM(128→64→32) → Dense(64→32→1)
- **용도**: 장기 의존성 기반 RUL 예측
- **에포크**: 100 (Early Stopping)

---

## 파일 구조

```
선박 추진축계 CBM 시스템
├── 한바다호+선박의+추진축계+모니터링+데이터.json (73GB)
├── HBD_추진축계_데이터정의서.xlsx
├── requirements.txt
├── README.md
│
├── cbm_preprocessing.py          # 데이터 전처리
├── cbm_anomaly_detection.py      # 이상 탐지 모델
├── cbm_predictive_maintenance.py # 예지 보전 모델
├── cbm_dashboard.py              # 실시간 대시보드
│
├── preprocessed_shaft_data.parquet   # 전처리 데이터
│
├── isolation_forest_model.pkl
├── autoencoder_model.h5
├── autoencoder_model_threshold.npy
├── lstm_autoencoder_model.h5
├── lstm_autoencoder_model_threshold.npy
├── random_forest_rul_model.pkl
├── gradient_boosting_rul_model.pkl
├── lstm_rul_model.h5
│
└── [시각화 결과 이미지들]
    ├── isolation_forest_anomaly_visualization.png
    ├── autoencoder_anomaly_visualization.png
    ├── lstm_autoencoder_anomaly_visualization.png
    ├── autoencoder_training_history.png
    ├── random_forest_rul_visualization.png
    ├── gradient_boosting_rul_visualization.png
    ├── lstm_rul_visualization.png
    ├── lstm_rul_training_history.png
    └── health_index_timeline.png
```

---

## 대시보드 기능

### 탭 1: 실시간 모니터링
- 4개 진동 센서 실시간 차트
- RPM 모니터링 + 이동평균
- 주요 메트릭 (RPM, 진동, 건강지수)

### 탭 2: 이상 탐지
- 3개 모델 통합 이상 탐지 결과
- 이상치 발생 히스토리
- 센서별 이상치 통계

### 탭 3: 예지 보전
- 건강 지수 게이지 차트
- RUL 게이지 차트
- 건강 지수 추이 그래프
- 유지보수 권장사항

### 탭 4: 통계 분석
- 센서별 기술 통계
- 특징 간 상관관계 히트맵
- 시간대별 진동 분석

---

## ⚙️ 커스터마이징

### 모델 하이퍼파라미터 조정

#### Isolation Forest
```python
# cbm_anomaly_detection.py
contamination=0.1  # 이상치 비율 (0.01~0.3)
n_estimators=100   # 트리 개수 (50~200)
```

#### Autoencoder
```python
# cbm_anomaly_detection.py
encoding_dim=32           # 잠재 차원 (16~64)
hidden_layers=[64, 32]    # 히든 레이어 구조
epochs=100                # 에포크 (50~200)
```

#### RUL 예측 - LSTM
```python
# cbm_predictive_maintenance.py
sequence_length=100   # 시퀀스 길이 (50~200)
epochs=100           # 에포크 (50~200)
```

### 건강 지수 임계값
```python
# cbm_predictive_maintenance.py
failure_threshold=30   # 고장 임계값 (20~40)
warning_threshold=70   # 경고 임계값 (60~80)
```

---

## 🔍 트러블슈팅

### 1. 메모리 부족
```python
# cbm_preprocessing.py에서 샘플 크기 조정
max_records=500000  # 100만 → 50만으로 감소
```

### 2. 학습 시간이 너무 길 경우
```python
# 에포크 수 감소
epochs=50  # 100 → 50

# 배치 크기 증가
batch_size=512  # 256 → 512
```

### 3. GPU 사용
```python
# tensorflow GPU 버전 설치
pip install tensorflow-gpu==2.13.0

# GPU 확인
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 📈 성능 평가 지표

### 이상 탐지
- **이상 탐지 비율**: 전체 데이터 대비 이상으로 판단된 비율
- **재구성 오차**: Autoencoder의 입력-출력 차이 (낮을수록 좋음)

### RUL 예측
- **RMSE** (Root Mean Squared Error): 예측 오차 (낮을수록 좋음)
- **MAE** (Mean Absolute Error): 평균 절대 오차 (낮을수록 좋음)
- **R²** (R-squared): 결정계수 (1에 가까울수록 좋음)

---

## 참고사항

1. **데이터 크기**: 원본 JSON 파일이 73GB로 매우 큽니다. 프로토타입 개발 시 샘플 데이터(100만 레코드)만 사용하는 것을 권장합니다.

2. **서버 배포**: 실제 서버 환경에서는 전체 데이터를 배치 처리하고, 학습된 모델만 대시보드에서 사용하세요.

3. **실시간 스트리밍**: 실제 운영 환경에서는 MQTT, Kafka 등을 활용한 실시간 데이터 파이프라인 구축을 권장합니다.

4. **모델 업데이트**: 주기적으로 새로운 데이터로 모델을 재학습하여 성능을 유지하세요.

---

## 기여

버그 리포트, 기능 제안, 코드 개선 등 모든 기여를 환영합니다!

---

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

---

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**한바다호 선박 추진축계 CBM 시스템을 사용해주셔서 감사합니다!**
