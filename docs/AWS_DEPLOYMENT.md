# 🚀 CBM Anomaly Detection Dashboard - AWS 배포 가이드

## 목차
1. [배포 방법 선택](#배포-방법-선택)
2. [Option 1: AWS App Runner (가장 간단)](#option-1-aws-app-runner)
3. [Option 2: AWS ECS Fargate (프로덕션 권장)](#option-2-aws-ecs-fargate)
4. [Option 3: AWS EC2 (완전한 제어)](#option-3-aws-ec2)
5. [공통 설정](#공통-설정)
6. [모니터링 및 관리](#모니터링-및-관리)

---

## 배포 방법 선택

| 방법 | 난이도 | 비용 | 확장성 | 추천 대상 |
|------|--------|------|--------|-----------|
| **AWS App Runner** | ⭐ 쉬움 | $ | 자동 | 빠른 프로토타입, POC |
| **AWS ECS Fargate** | ⭐⭐ 중간 | $$ | 우수 | **프로덕션 추천** |
| **AWS EC2** | ⭐⭐⭐ 어려움 | $$$ | 수동 | 완전한 제어 필요 시 |

---

## Option 1: AWS App Runner

### 🌟 장점
- 가장 간단하고 빠른 배포
- 자동 스케일링
- HTTPS 자동 설정
- GitHub 직접 연동 가능

### 📋 전제 조건
```bash
# AWS CLI 설치 및 설정
aws configure
# AWS Access Key ID: [입력]
# AWS Secret Access Key: [입력]
# Default region: ap-northeast-2
```

### 1️⃣ Dockerfile 생성

`D:\Downloads\Dockerfile` 파일 생성:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Streamlit 포트 노출
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### 2️⃣ requirements.txt 생성

```txt
streamlit==1.40.1
plotly==5.24.0
mlflow==2.9.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.2.2
pyarrow==17.0.0
```

### 3️⃣ Docker 이미지 빌드 및 푸시

```bash
# Docker 이미지 빌드
cd D:\Downloads
docker build -t cbm-dashboard:latest .

# Docker 테스트 (로컬)
docker run -p 8501:8501 cbm-dashboard:latest

# ECR 생성
aws ecr create-repository --repository-name cbm-dashboard --region ap-northeast-2

# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com

# 이미지 태그 및 푸시
docker tag cbm-dashboard:latest <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/cbm-dashboard:latest
docker push <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/cbm-dashboard:latest
```

### 4️⃣ App Runner 서비스 생성

```bash
# apprunner.yaml 생성
cat > apprunner.yaml << 'EOF'
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
  network:
    port: 8501
EOF

# App Runner 서비스 생성
aws apprunner create-service \\
  --service-name cbm-dashboard \\
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/cbm-dashboard:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8501"
      }
    },
    "AutoDeploymentsEnabled": true
  }' \\
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \\
  --region ap-northeast-2
```

### 5️⃣ 접속 확인

```bash
# 서비스 URL 확인
aws apprunner list-services --region ap-northeast-2

# 출력 예시:
# https://xxxxx.ap-northeast-2.awsapprunner.com
```

---

## Option 2: AWS ECS Fargate (프로덕션 권장)

### 🌟 장점
- 서버리스 컨테이너 실행
- 자동 스케일링
- Application Load Balancer 통합
- VPC 내 보안 강화
- 비용 효율적

### 📋 아키텍처

```
[사용자] → [ALB] → [ECS Fargate Task] → [RDS PostgreSQL]
                            ↓
                    [S3 (모델 저장)]
                            ↓
                    [CloudWatch (로그)]
```

### 1️⃣ VPC 및 네트워크 설정

```bash
# VPC 생성
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --region ap-northeast-2

# 서브넷 생성 (Public x2, Private x2)
aws ec2 create-subnet --vpc-id vpc-xxxxx --cidr-block 10.0.1.0/24 --availability-zone ap-northeast-2a
aws ec2 create-subnet --vpc-id vpc-xxxxx --cidr-block 10.0.2.0/24 --availability-zone ap-northeast-2b
aws ec2 create-subnet --vpc-id vpc-xxxxx --cidr-block 10.0.3.0/24 --availability-zone ap-northeast-2a
aws ec2 create-subnet --vpc-id vpc-xxxxx --cidr-block 10.0.4.0/24 --availability-zone ap-northeast-2b

# Internet Gateway 생성 및 연결
aws ec2 create-internet-gateway
aws ec2 attach-internet-gateway --vpc-id vpc-xxxxx --internet-gateway-id igw-xxxxx
```

### 2️⃣ ECS 클러스터 생성

```bash
# ECS 클러스터 생성
aws ecs create-cluster --cluster-name cbm-production --region ap-northeast-2
```

### 3️⃣ Task Definition 생성

`task-definition.json` 파일 생성:

```json
{
  "family": "cbm-dashboard",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "cbm-dashboard",
      "image": "<account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/cbm-dashboard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MLFLOW_RUN_ID",
          "value": "c66eb2b2ae5e4057b839264080ce24e5"
        },
        {
          "name": "AWS_REGION",
          "value": "ap-northeast-2"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cbm-dashboard",
          "awslogs-region": "ap-northeast-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

```bash
# Task Definition 등록
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### 4️⃣ Application Load Balancer 생성

```bash
# ALB 생성
aws elbv2 create-load-balancer \\
  --name cbm-alb \\
  --subnets subnet-xxxxx subnet-yyyyy \\
  --security-groups sg-xxxxx \\
  --region ap-northeast-2

# Target Group 생성
aws elbv2 create-target-group \\
  --name cbm-targets \\
  --protocol HTTP \\
  --port 8501 \\
  --vpc-id vpc-xxxxx \\
  --target-type ip \\
  --health-check-path /_stcore/health \\
  --region ap-northeast-2

# Listener 생성
aws elbv2 create-listener \\
  --load-balancer-arn arn:aws:elasticloadbalancing:... \\
  --protocol HTTP \\
  --port 80 \\
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### 5️⃣ ECS 서비스 생성

```bash
# ECS 서비스 생성
aws ecs create-service \\
  --cluster cbm-production \\
  --service-name cbm-dashboard-service \\
  --task-definition cbm-dashboard:1 \\
  --desired-count 2 \\
  --launch-type FARGATE \\
  --network-configuration '{
    "awsvpcConfiguration": {
      "subnets": ["subnet-xxxxx", "subnet-yyyyy"],
      "securityGroups": ["sg-xxxxx"],
      "assignPublicIp": "ENABLED"
    }
  }' \\
  --load-balancers '[{
    "targetGroupArn": "arn:aws:elasticloadbalancing:...",
    "containerName": "cbm-dashboard",
    "containerPort": 8501
  }]' \\
  --region ap-northeast-2
```

### 6️⃣ Auto Scaling 설정

```bash
# Auto Scaling Target 등록
aws application-autoscaling register-scalable-target \\
  --service-namespace ecs \\
  --scalable-dimension ecs:service:DesiredCount \\
  --resource-id service/cbm-production/cbm-dashboard-service \\
  --min-capacity 2 \\
  --max-capacity 10

# Auto Scaling Policy 생성 (CPU 기반)
aws application-autoscaling put-scaling-policy \\
  --service-namespace ecs \\
  --scalable-dimension ecs:service:DesiredCount \\
  --resource-id service/cbm-production/cbm-dashboard-service \\
  --policy-name cpu-scaling \\
  --policy-type TargetTrackingScaling \\
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

---

## Option 3: AWS EC2

### 1️⃣ EC2 인스턴스 생성

```bash
# EC2 인스턴스 시작 (Ubuntu 22.04 LTS)
aws ec2 run-instances \\
  --image-id ami-0c9c942bd7bf113a2 \\
  --instance-type t3.medium \\
  --key-name your-key-pair \\
  --security-group-ids sg-xxxxx \\
  --subnet-id subnet-xxxxx \\
  --region ap-northeast-2
```

### 2️⃣ EC2에 SSH 접속 및 설정

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.ap-northeast-2.compute.amazonaws.com

# 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# Python 3.11 설치
sudo apt install python3.11 python3.11-venv python3-pip -y

# 프로젝트 디렉토리 생성
mkdir cbm-dashboard
cd cbm-dashboard

# 파일 업로드 (로컬에서 실행)
# scp -i your-key.pem -r D:\Downloads/* ubuntu@ec2-xx-xx-xx-xx:/home/ubuntu/cbm-dashboard/

# 가상환경 생성
python3.11 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# Nginx 설치 (리버스 프록시)
sudo apt install nginx -y

# Nginx 설정
sudo nano /etc/nginx/sites-available/cbm-dashboard
```

Nginx 설정 내용:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Nginx 활성화
sudo ln -s /etc/nginx/sites-available/cbm-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Systemd 서비스 생성
sudo nano /etc/systemd/system/cbm-dashboard.service
```

Systemd 서비스 내용:

```ini
[Unit]
Description=CBM Anomaly Detection Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cbm-dashboard
Environment="PATH=/home/ubuntu/cbm-dashboard/venv/bin"
ExecStart=/home/ubuntu/cbm-dashboard/venv/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable cbm-dashboard
sudo systemctl start cbm-dashboard
sudo systemctl status cbm-dashboard
```

---

## 공통 설정

### 1️⃣ S3에 모델 저장

```bash
# S3 버킷 생성
aws s3 mb s3://cbm-anomaly-models --region ap-northeast-2

# 모델 업로드
aws s3 sync D:\Downloads\mlruns\ s3://cbm-anomaly-models/mlruns/ --region ap-northeast-2

# config.py 수정하여 S3에서 모델 로드하도록 변경
```

`config.py` 수정:

```python
import boto3
import mlflow

# S3에서 모델 로드
mlflow.set_tracking_uri("s3://cbm-anomaly-models/mlruns")
MLFLOW_RUN_ID = "c66eb2b2ae5e4057b839264080ce24e5"
```

### 2️⃣ RDS PostgreSQL 설정 (프로덕션용)

```bash
# RDS 인스턴스 생성
aws rds create-db-instance \\
  --db-instance-identifier cbm-database \\
  --db-instance-class db.t3.micro \\
  --engine postgres \\
  --master-username admin \\
  --master-user-password YourPassword123! \\
  --allocated-storage 20 \\
  --region ap-northeast-2
```

### 3️⃣ CloudWatch 로그 설정

```bash
# CloudWatch Logs 그룹 생성
aws logs create-log-group --log-group-name /ecs/cbm-dashboard --region ap-northeast-2

# 로그 보존 기간 설정 (30일)
aws logs put-retention-policy \\
  --log-group-name /ecs/cbm-dashboard \\
  --retention-in-days 30 \\
  --region ap-northeast-2
```

---

## 모니터링 및 관리

### 1️⃣ CloudWatch 대시보드 생성

```bash
# 대시보드 생성
aws cloudwatch put-dashboard \\
  --dashboard-name CBM-Dashboard \\
  --dashboard-body file://dashboard.json
```

`dashboard.json`:

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", {"stat": "Average"}],
          [".", "MemoryUtilization", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "ap-northeast-2",
        "title": "ECS Resource Usage"
      }
    }
  ]
}
```

### 2️⃣ 알람 설정

```bash
# CPU 사용률 알람
aws cloudwatch put-metric-alarm \\
  --alarm-name cbm-high-cpu \\
  --alarm-description "Alert when CPU exceeds 80%" \\
  --metric-name CPUUtilization \\
  --namespace AWS/ECS \\
  --statistic Average \\
  --period 300 \\
  --threshold 80 \\
  --comparison-operator GreaterThanThreshold \\
  --evaluation-periods 2 \\
  --region ap-northeast-2
```

### 3️⃣ 배포 자동화 (CI/CD with GitHub Actions)

`.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-2

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: cbm-dashboard
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Update ECS service
        run: |
          aws ecs update-service \\
            --cluster cbm-production \\
            --service cbm-dashboard-service \\
            --force-new-deployment
```

---

## 보안 고려사항

### 1️⃣ IAM 역할 생성

```bash
# ECS Task Execution Role
aws iam create-role \\
  --role-name ecsTaskExecutionRole \\
  --assume-role-policy-document file://trust-policy.json

# S3, ECR, CloudWatch 권한 추가
aws iam attach-role-policy \\
  --role-name ecsTaskExecutionRole \\
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### 2️⃣ Secrets Manager 사용

```bash
# 비밀번호 저장
aws secretsmanager create-secret \\
  --name cbm/database/password \\
  --secret-string "YourSecurePassword123!" \\
  --region ap-northeast-2
```

### 3️⃣ VPC Security Groups

```bash
# ALB 보안 그룹 (HTTP/HTTPS 허용)
aws ec2 create-security-group \\
  --group-name cbm-alb-sg \\
  --description "Security group for ALB" \\
  --vpc-id vpc-xxxxx

aws ec2 authorize-security-group-ingress \\
  --group-id sg-xxxxx \\
  --protocol tcp \\
  --port 80 \\
  --cidr 0.0.0.0/0

# ECS 보안 그룹 (ALB에서만 허용)
aws ec2 create-security-group \\
  --group-name cbm-ecs-sg \\
  --description "Security group for ECS tasks" \\
  --vpc-id vpc-xxxxx

aws ec2 authorize-security-group-ingress \\
  --group-id sg-yyyyy \\
  --protocol tcp \\
  --port 8501 \\
  --source-group sg-xxxxx
```

---

## 비용 최적화

### 1️⃣ Reserved Instances (EC2)
- 1년/3년 예약 시 최대 72% 절감

### 2️⃣ Fargate Spot (ECS)
- Spot Fargate 사용 시 최대 70% 절감

### 3️⃣ S3 Intelligent-Tiering
- 자동으로 저렴한 스토리지 클래스로 이동

### 4️⃣ CloudWatch Logs 보존 기간 설정
- 불필요한 로그 자동 삭제

---

## 예상 비용

### App Runner
- **$25-50/월** (vCPU 1, Memory 2GB, 낮은 트래픽)

### ECS Fargate
- **$50-100/월** (Task 2개, vCPU 1, Memory 2GB)
- ALB: ~$20/월
- **총 $70-120/월**

### EC2
- **t3.medium: $35/월** (On-Demand)
- EBS: ~$10/월
- **총 $45/월** (단, 수동 관리 필요)

---

## 문제 해결

### 1️⃣ 컨테이너가 시작하지 않음
```bash
# 로그 확인
aws logs tail /ecs/cbm-dashboard --follow --region ap-northeast-2
```

### 2️⃣ Health Check 실패
- `/_stcore/health` 엔드포인트 확인
- 시작 시간 증가: `startPeriod: 120`

### 3️⃣ 메모리 부족
- Task Definition에서 메모리 증가: `"memory": "4096"`

---

## 다음 단계

1. ✅ **HTTPS 설정**: AWS Certificate Manager + CloudFront
2. ✅ **인증/권한**: AWS Cognito 통합
3. ✅ **데이터베이스**: RDS PostgreSQL 연결
4. ✅ **캐싱**: ElastiCache Redis
5. ✅ **API Gateway**: RESTful API 추가

---

## 참고 자료

- [AWS App Runner 공식 문서](https://docs.aws.amazon.com/apprunner/)
- [AWS ECS 공식 문서](https://docs.aws.amazon.com/ecs/)
- [Streamlit 배포 가이드](https://docs.streamlit.io/deploy)
- [MLflow on AWS](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)

---

**작성일**: 2025-11-05
**버전**: 1.0
**지원**: cbm-support@example.com
