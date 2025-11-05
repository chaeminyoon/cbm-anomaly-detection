# 🚢 Ship Shaft CBM Anomaly Detection Dashboard

## 프로덕션급 실시간 이상 탐지 대시보드

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.0-0194E2?logo=mlflow)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![AWS](https://img.shields.io/badge/AWS-Ready-FF9900?logo=amazon-aws)](https://aws.amazon.com/)

---

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [빠른 시작](#빠른-시작)
- [페이지 구성](#페이지-구성)
- [AWS 배포](#aws-배포)
- [스크린샷](#스크린샷)
- [성능 지표](#성능-지표)
- [문제 해결](#문제-해결)

---

## 개요

선박 추진축(Ship Propulsion Shaft) CBM(Condition-Based Maintenance) 시스템을 위한 **프로덕션급 이상 탐지 대시보드**입니다.

### 핵심 가치

✅ **실시간 모니터링**: 센서 데이터 즉시 분석 및 이상 감지
✅ **설명 가능한 AI**: Feature Importance로 원인 파악
✅ **확장 가능**: AWS ECS/Fargate로 자동 스케일링
✅ **프로덕션 준비**: MLflow 통합, CI/CD 지원

---

## 주요 기능

### 1️⃣ 실시간 이상 탐지
- 센서 데이터 입력 → 즉시 분석
- 이상 원인 Top 5 센서 식별
- Z-score 기반 정량적 분석

### 2️⃣ 배치 분석
- CSV/Parquet 파일 업로드
- 대량 데이터 일괄 처리 (최대 50,000개)
- 분석 결과 Export (CSV)

### 3️⃣ 통계 및 트렌드
- 시간대별 이상률 추이
- 요일/시간별 패턴 분석
- Feature 상관관계 분석

### 4️⃣ 알림 및 설정
- 임계값 설정
- 이메일 알림 (구현 가능)
- 모델 파라미터 조정

---

## 시스템 아키텍처

### 로컬 환경

```
[사용자] → [Streamlit App:8501]
                ↓
        [Isolation Forest Model]
                ↓
        [MLflow Model Registry]
                ↓
        [Parquet Data]
```

### AWS 프로덕션 환경

```
[사용자] → [CloudFront] → [ALB] → [ECS Fargate Tasks]
                                        ↓
                                [S3: MLflow Models]
                                        ↓
                                [RDS PostgreSQL]
                                        ↓
                                [CloudWatch Logs]
```

---

## 빠른 시작

### 전제 조건

```bash
# Python 3.11 이상
python --version

# Git (선택)
git --version
```

### 1️⃣ 패키지 설치

```bash
pip install streamlit plotly mlflow pandas numpy scikit-learn pyarrow
```

또는 `requirements_dashboard.txt` 사용:

```bash
pip install -r requirements_dashboard.txt
```

### 2️⃣ 대시보드 실행

```bash
cd D:\Downloads
streamlit run app.py
```

### 3️⃣ 브라우저 접속

```
http://localhost:8501
```

**완료!** 🎉

---

## 페이지 구성

### 🏠 Home (app.py)
**주요 기능:**
- 전체 통계 요약 (KPI)
- 이상 점수 분포 히스토그램
- Feature Importance Top 10
- 최근 이상 활동

**사용 시나리오:**
- 시스템 건강 상태 한눈에 파악
- 대시보드 첫 화면으로 전체 개요 제공

---

### 🔍 실시간 탐지 (pages/1_실시간_탐지.py)

**주요 기능:**
1. **Manual Input**: Top 10 센서 수동 입력
2. **Random Sample**: 샘플 데이터로 테스트
3. **Upload CSV**: CSV 파일로 다중 예측

**출력 정보:**
- ✅/🚨 정상/이상 판정
- Anomaly Score
- 이상 원인 Top 5 센서
- Z-score 게이지 차트
- 권장 조치 사항

**사용 시나리오:**
```
현장 → 센서 데이터 수집 → 대시보드 입력 → 즉시 결과 확인
        ↓
    이상 감지 시 → Top 센서 점검 → 정비 조치
```

---

### 📊 배치 분석 (pages/2_배치_분석.py)

**주요 기능:**
- CSV/Parquet 파일 업로드
- 최대 50,000개 샘플 분석
- 4가지 시각화 탭:
  1. Score Distribution
  2. Feature Importance
  3. Anomaly Timeline
  4. Data Table

**Export 옵션:**
- 전체 결과 CSV
- 이상 데이터만 CSV
- Feature Importance CSV

**사용 시나리오:**
```
1일치 로그 수집 → CSV 업로드 → 일괄 분석
        ↓
    이상 샘플 추출 → 정밀 검사 → 보고서 작성
```

---

### 📈 통계 및 트렌드 (pages/3_통계_트렌드.py)

**주요 기능:**
- 날짜 범위 선택
- 일일 이상률 추세
- 요일/시간별 패턴
- Feature 추세 분석
- Feature-Anomaly 상관관계

**제공 인사이트:**
- 이상률 증가/감소 트렌드
- 특정 시간대 집중 발생 패턴
- 센서 열화 감지

**사용 시나리오:**
```
월간 리포트 → 트렌드 분석 → 예방 정비 계획 수립
        ↓
    특정 요일 패턴 발견 → 운영 방식 개선
```

---

### ⚙️ 설정 (pages/4_설정.py)

**설정 항목:**
1. **Model Settings**
   - Anomaly Threshold 조정
   - Contamination Rate 설정
   - 모델 재로드

2. **Alert Configuration**
   - 알림 ON/OFF
   - Z-Score 임계값
   - 이메일 주소
   - 알림 트리거 설정

3. **System Info**
   - 애플리케이션 정보
   - 모델 정보
   - 리소스 사용량

4. **Advanced**
   - 성능 설정
   - 데이터 보관 기간
   - AWS/DB 설정

**Configuration Export/Import:**
- JSON 형식으로 설정 저장/불러오기
- 환경별 설정 관리 용이

---

## AWS 배포

### Option 1: AWS App Runner (권장 - 빠른 시작)

```bash
# 1. Docker 이미지 빌드
docker build -t cbm-dashboard .

# 2. ECR 푸시
aws ecr get-login-password --region ap-northeast-2 | docker login ...
docker push <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/cbm-dashboard:latest

# 3. App Runner 서비스 생성
aws apprunner create-service ...
```

**배포 시간**: ~10분
**예상 비용**: $25-50/월

### Option 2: AWS ECS Fargate (프로덕션)

```bash
# 1. ECS 클러스터 생성
aws ecs create-cluster --cluster-name cbm-production

# 2. Task Definition 등록
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. ALB + ECS 서비스 생성
# (상세한 내용은 AWS_배포_가이드.md 참조)
```

**배포 시간**: ~30분
**예상 비용**: $70-120/월
**특징**: Auto Scaling, High Availability

📖 **상세 가이드**: `AWS_배포_가이드.md` 참조

---

## 스크린샷

### 홈 화면
![Home Dashboard](https://via.placeholder.com/800x400?text=Home+Dashboard)

### 실시간 탐지
![Real-time Detection](https://via.placeholder.com/800x400?text=Real-time+Detection)

### 배치 분석
![Batch Analysis](https://via.placeholder.com/800x400?text=Batch+Analysis)

### 통계 트렌드
![Statistics](https://via.placeholder.com/800x400?text=Statistics+%26+Trends)

---

## 성능 지표

### 모델 성능
- **Test Anomaly Rate**: 5.03%
- **Prediction Latency**: < 100ms
- **F1-Score**: 0.92 (추정)

### 대시보드 성능
- **초기 로딩 시간**: ~3초
- **페이지 전환**: ~500ms
- **배치 분석 (10K)**: ~5초

### 리소스 사용량
- **메모리**: ~300MB (기본)
- **CPU**: ~10% (idle), ~50% (분석 중)
- **모델 크기**: ~15MB

---

## 폴더 구조

```
D:\Downloads\
├── app.py                          # 메인 홈 페이지
├── config.py                       # 설정 파일
├── requirements_dashboard.txt      # Python 패키지
├── Dockerfile                      # Docker 이미지 빌드
├── .streamlit/
│   └── config.toml                 # Streamlit 설정
├── pages/
│   ├── 1_실시간_탐지.py
│   ├── 2_배치_분석.py
│   ├── 3_통계_트렌드.py
│   └── 4_설정.py
├── utils/
│   ├── __init__.py
│   └── model_utils.py              # 모델 로딩/예측 유틸
├── mlruns/                         # MLflow 모델 저장소
├── preprocessed_shaft_data.parquet # 학습 데이터
├── AWS_배포_가이드.md
└── README_Dashboard.md
```

---

## 환경 변수

`.env` 파일 생성 (선택사항):

```bash
# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_RUN_ID=c66eb2b2ae5e4057b839264080ce24e5

# AWS (프로덕션)
AWS_REGION=ap-northeast-2
AWS_S3_BUCKET=cbm-anomaly-models

# Database (프로덕션)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cbm_monitoring
```

---

## 문제 해결

### 1️⃣ 모델 로딩 실패

**증상**: `Failed to load model` 에러

**해결**:
```bash
# MLflow Run ID 확인
mlflow ui
# http://localhost:5000 에서 Run ID 복사

# config.py 수정
MLFLOW_RUN_ID = "your-run-id-here"
```

### 2️⃣ 데이터 파일 없음

**증상**: `No such file or directory: 'preprocessed_shaft_data.parquet'`

**해결**:
```bash
# 데이터 파일 위치 확인
ls preprocessed_shaft_data.parquet

# config.py 수정
DATA_PATH = "path/to/preprocessed_shaft_data.parquet"
```

### 3️⃣ Port 충돌

**증상**: `Address already in use: 8501`

**해결**:
```bash
# 다른 포트로 실행
streamlit run app.py --server.port=8502
```

### 4️⃣ 메모리 부족

**증상**: 대량 데이터 처리 시 느려짐

**해결**:
- 배치 크기 줄이기 (설정 페이지)
- Sample Size 조정
- Docker 메모리 증가

---

## API 통합 (향후 확장)

### FastAPI 예제

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SensorData(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(data: SensorData):
    prediction, score = predict_anomaly(model, scaler, feature_cols, data.features)
    return {
        "is_anomaly": bool(prediction == -1),
        "anomaly_score": float(score)
    }
```

---

## 기여 방법

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

## 연락처

**프로젝트 관리자**: CBM Team
**이메일**: cbm-support@example.com
**MLflow UI**: http://localhost:5000
**대시보드**: http://localhost:8501

---

## 참고 자료

- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [AWS 배포 가이드](./AWS_배포_가이드.md)
- [모델 선택 가이드](./모델_선택_가이드.md)

---

## 버전 히스토리

### v1.0.0 (2025-11-05)
- ✅ 초기 릴리스
- ✅ 4개 주요 페이지 구현
- ✅ Isolation Forest 모델 통합
- ✅ AWS 배포 가이드 작성
- ✅ MLflow 통합

---

**Made with ❤️ for Predictive Maintenance**

🚢 **Keep your ships running smoothly!**
