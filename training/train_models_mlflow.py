"""
CBM 이상 탐지 시스템 - MLflow 통합 버전
추진축계 데이터를 사용한 이상 탐지 모델 학습 및 추적

MLflow를 사용하여 실험, 파라미터, 메트릭, 모델을 자동으로 추적합니다.
"""

# Windows 환경에서 UTF-8 인코딩 설정
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import mlflow
import mlflow.keras
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


class AnomalyDetectionMLflow:
    """MLflow를 활용한 이상 탐지 시스템"""

    def __init__(self, data_path='preprocessed_shaft_data.parquet', experiment_name='CBM_Shaft_Anomaly_Detection'):
        """
        초기화

        Args:
            data_path: 전처리된 데이터 경로
            experiment_name: MLflow 실험 이름
        """
        self.data_path = data_path
        self.experiment_name = experiment_name

        # MLflow 실험 설정 (sqlite 백엔드 — MLflow 2.x/3.x 공통으로 동작하도록 명시)
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow 실험 설정: {experiment_name}")

        # 데이터 로드
        self.load_data()

    def load_data(self):
        """전처리된 데이터 로드"""
        print(f"\n📂 데이터 로드 중: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        print(f"   데이터 shape: {self.df.shape}")

        # 특징 컬럼 (수치형 데이터만)
        self.feature_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in self.feature_cols:
            self.feature_cols.remove('timestamp')

        print(f"   특징 컬럼 수: {len(self.feature_cols)}")

    def prepare_data(self, test_size=0.2, random_state=42):
        """데이터 준비 및 정규화"""
        print(f"\n🔧 데이터 준비 중 (test_size={test_size})")

        # 결측치 제거
        self.df = self.df.dropna(subset=self.feature_cols)

        # Train/Test 분할
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state
        )

        # 정규화
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(train_df[self.feature_cols])
        self.X_test = self.scaler.transform(test_df[self.feature_cols])

        print(f"   Train shape: {self.X_train.shape}")
        print(f"   Test shape: {self.X_test.shape}")

        return self.X_train, self.X_test

    def train_autoencoder(self, encoding_dim=32, epochs=50, batch_size=256):
        """
        Autoencoder 모델 학습 (MLflow 추적)

        Args:
            encoding_dim: 인코딩 차원
            epochs: 학습 에포크
            batch_size: 배치 크기
        """
        print(f"\n🚀 Autoencoder 모델 학습 시작")

        with mlflow.start_run(run_name="Autoencoder") as run:
            # 파라미터 로깅
            params = {
                "model_type": "Autoencoder",
                "encoding_dim": encoding_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "input_dim": self.X_train.shape[1],
                "optimizer": "adam",
                "loss": "mse"
            }
            mlflow.log_params(params)

            # 모델 구성
            input_dim = self.X_train.shape[1]
            input_layer = Input(shape=(input_dim,))

            # Encoder
            encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
            encoded = Dense(encoding_dim, activation='relu')(encoded)

            # Decoder
            decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='linear')(decoded)

            # 모델 생성
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            # Early Stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # 학습
            history = autoencoder.fit(
                self.X_train, self.X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1
            )

            # 메트릭 로깅
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            mlflow.log_metric("final_train_loss", final_train_loss)
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("epochs_trained", len(history.history['loss']))

            # 재구성 오차 계산
            train_pred = autoencoder.predict(self.X_train, verbose=0)
            test_pred = autoencoder.predict(self.X_test, verbose=0)

            train_mse = np.mean(np.power(self.X_train - train_pred, 2), axis=1)
            test_mse = np.mean(np.power(self.X_test - test_pred, 2), axis=1)

            # Threshold 계산 (95 percentile)
            threshold = np.percentile(train_mse, 95)
            mlflow.log_metric("threshold", threshold)

            # 이상치 탐지
            train_anomalies = train_mse > threshold
            test_anomalies = test_mse > threshold

            train_anomaly_rate = np.mean(train_anomalies) * 100
            test_anomaly_rate = np.mean(test_anomalies) * 100

            mlflow.log_metric("train_anomaly_rate", train_anomaly_rate)
            mlflow.log_metric("test_anomaly_rate", test_anomaly_rate)

            print(f"   ✅ 학습 완료!")
            print(f"   - Threshold: {threshold:.6f}")
            print(f"   - Train 이상 비율: {train_anomaly_rate:.2f}%")
            print(f"   - Test 이상 비율: {test_anomaly_rate:.2f}%")

            # 시각화
            self._plot_autoencoder_results(
                history, train_mse, test_mse, threshold,
                train_anomalies, test_anomalies
            )

            # 모델 저장
            mlflow.keras.log_model(autoencoder, "model")

            # Threshold 저장
            np.save('autoencoder_threshold.npy', threshold)
            mlflow.log_artifact('autoencoder_threshold.npy')

            # 시각화 저장
            mlflow.log_artifact('autoencoder_training_history.png')
            mlflow.log_artifact('autoencoder_anomaly_visualization.png')

            # Run ID 저장
            self.autoencoder_run_id = run.info.run_id
            print(f"   📊 MLflow Run ID: {self.autoencoder_run_id}")

            return autoencoder, threshold

    def train_isolation_forest(self, contamination=0.05, n_estimators=100):
        """
        Isolation Forest 모델 학습 (MLflow 추적)

        Args:
            contamination: 이상치 비율
            n_estimators: 트리 개수
        """
        print(f"\n🌲 Isolation Forest 모델 학습 시작")

        with mlflow.start_run(run_name="IsolationForest") as run:
            # 파라미터 로깅
            params = {
                "model_type": "IsolationForest",
                "contamination": contamination,
                "n_estimators": n_estimators,
                "max_samples": "auto",
                "random_state": 42
            }
            mlflow.log_params(params)

            # 모델 생성 및 학습
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )

            iso_forest.fit(self.X_train)

            # 예측
            train_pred = iso_forest.predict(self.X_train)
            test_pred = iso_forest.predict(self.X_test)

            # 이상 점수
            train_scores = iso_forest.score_samples(self.X_train)
            test_scores = iso_forest.score_samples(self.X_test)

            # 이상치 탐지 (-1: 이상, 1: 정상)
            train_anomalies = train_pred == -1
            test_anomalies = test_pred == -1

            train_anomaly_rate = np.mean(train_anomalies) * 100
            test_anomaly_rate = np.mean(test_anomalies) * 100

            # 메트릭 로깅
            mlflow.log_metric("train_anomaly_rate", train_anomaly_rate)
            mlflow.log_metric("test_anomaly_rate", test_anomaly_rate)
            mlflow.log_metric("mean_train_score", np.mean(train_scores))
            mlflow.log_metric("mean_test_score", np.mean(test_scores))

            print(f"   ✅ 학습 완료!")
            print(f"   - Train 이상 비율: {train_anomaly_rate:.2f}%")
            print(f"   - Test 이상 비율: {test_anomaly_rate:.2f}%")

            # 시각화
            self._plot_isolation_forest_results(
                train_scores, test_scores,
                train_anomalies, test_anomalies
            )

            # 모델 저장
            mlflow.sklearn.log_model(iso_forest, "model")

            # Pickle 저장
            with open('isolation_forest_model.pkl', 'wb') as f:
                pickle.dump(iso_forest, f)
            mlflow.log_artifact('isolation_forest_model.pkl')

            # 시각화 저장
            mlflow.log_artifact('isolation_forest_anomaly_visualization.png')

            # Run ID 저장
            self.isolation_forest_run_id = run.info.run_id
            print(f"   📊 MLflow Run ID: {self.isolation_forest_run_id}")

            return iso_forest

    def train_lstm_autoencoder(self, encoding_dim=32, epochs=50, batch_size=256, timesteps=10):
        """
        LSTM Autoencoder 모델 학습 (MLflow 추적)

        Args:
            encoding_dim: 인코딩 차원
            epochs: 학습 에포크
            batch_size: 배치 크기
            timesteps: 시계열 윈도우 크기
        """
        print(f"\n🔄 LSTM Autoencoder 모델 학습 시작")

        with mlflow.start_run(run_name="LSTM_Autoencoder") as run:
            # 시계열 데이터 준비
            X_train_lstm = self._create_sequences(self.X_train, timesteps)
            X_test_lstm = self._create_sequences(self.X_test, timesteps)

            # 파라미터 로깅
            params = {
                "model_type": "LSTM_Autoencoder",
                "encoding_dim": encoding_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "timesteps": timesteps,
                "input_dim": X_train_lstm.shape[2],
                "optimizer": "adam",
                "loss": "mse"
            }
            mlflow.log_params(params)

            # 모델 구성
            input_dim = X_train_lstm.shape[2]

            # Encoder
            inputs = Input(shape=(timesteps, input_dim))
            encoded = LSTM(encoding_dim, activation='relu', return_sequences=False)(inputs)

            # Decoder
            decoded = RepeatVector(timesteps)(encoded)
            decoded = LSTM(encoding_dim, activation='relu', return_sequences=True)(decoded)
            decoded = TimeDistributed(Dense(input_dim))(decoded)

            # 모델 생성
            lstm_autoencoder = Model(inputs=inputs, outputs=decoded)
            lstm_autoencoder.compile(optimizer='adam', loss='mse')

            # Early Stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # 학습
            history = lstm_autoencoder.fit(
                X_train_lstm, X_train_lstm,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1
            )

            # 메트릭 로깅
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            mlflow.log_metric("final_train_loss", final_train_loss)
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("epochs_trained", len(history.history['loss']))

            # 재구성 오차 계산
            train_pred = lstm_autoencoder.predict(X_train_lstm, verbose=0)
            test_pred = lstm_autoencoder.predict(X_test_lstm, verbose=0)

            train_mse = np.mean(np.power(X_train_lstm - train_pred, 2), axis=(1, 2))
            test_mse = np.mean(np.power(X_test_lstm - test_pred, 2), axis=(1, 2))

            # Threshold 계산
            threshold = np.percentile(train_mse, 95)
            mlflow.log_metric("threshold", threshold)

            # 이상치 탐지
            train_anomalies = train_mse > threshold
            test_anomalies = test_mse > threshold

            train_anomaly_rate = np.mean(train_anomalies) * 100
            test_anomaly_rate = np.mean(test_anomalies) * 100

            mlflow.log_metric("train_anomaly_rate", train_anomaly_rate)
            mlflow.log_metric("test_anomaly_rate", test_anomaly_rate)

            print(f"   ✅ 학습 완료!")
            print(f"   - Threshold: {threshold:.6f}")
            print(f"   - Train 이상 비율: {train_anomaly_rate:.2f}%")
            print(f"   - Test 이상 비율: {test_anomaly_rate:.2f}%")

            # 시각화
            self._plot_lstm_autoencoder_results(
                history, train_mse, test_mse, threshold,
                train_anomalies, test_anomalies
            )

            # 모델 저장
            mlflow.keras.log_model(lstm_autoencoder, "model")

            # Threshold 저장
            np.save('lstm_autoencoder_threshold.npy', threshold)
            mlflow.log_artifact('lstm_autoencoder_threshold.npy')

            # 시각화 저장
            mlflow.log_artifact('lstm_autoencoder_anomaly_visualization.png')

            # Run ID 저장
            self.lstm_autoencoder_run_id = run.info.run_id
            print(f"   📊 MLflow Run ID: {self.lstm_autoencoder_run_id}")

            return lstm_autoencoder, threshold

    def _create_sequences(self, data, timesteps):
        """시계열 시퀀스 생성"""
        sequences = []
        for i in range(len(data) - timesteps + 1):
            sequences.append(data[i:i+timesteps])
        return np.array(sequences)

    def _plot_autoencoder_results(self, history, train_mse, test_mse, threshold,
                                   train_anomalies, test_anomalies):
        """Autoencoder 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 학습 히스토리
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Autoencoder 학습 히스토리')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 재구성 오차 분포
        axes[0, 1].hist(train_mse, bins=50, alpha=0.7, label='Train')
        axes[0, 1].hist(test_mse, bins=50, alpha=0.7, label='Test')
        axes[0, 1].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        axes[0, 1].set_xlabel('재구성 오차 (MSE)')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('재구성 오차 분포')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Train 이상치 탐지
        axes[1, 0].scatter(range(len(train_mse)), train_mse, c=train_anomalies,
                          cmap='coolwarm', alpha=0.6, s=1)
        axes[1, 0].axhline(threshold, color='r', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('샘플 인덱스')
        axes[1, 0].set_ylabel('재구성 오차')
        axes[1, 0].set_title('Train 이상치 탐지')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Test 이상치 탐지
        axes[1, 1].scatter(range(len(test_mse)), test_mse, c=test_anomalies,
                          cmap='coolwarm', alpha=0.6, s=1)
        axes[1, 1].axhline(threshold, color='r', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('샘플 인덱스')
        axes[1, 1].set_ylabel('재구성 오차')
        axes[1, 1].set_title('Test 이상치 탐지')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('autoencoder_anomaly_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 학습 히스토리만 별도 저장
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder 학습 히스토리')
        plt.legend()
        plt.grid(True)
        plt.savefig('autoencoder_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_isolation_forest_results(self, train_scores, test_scores,
                                       train_anomalies, test_anomalies):
        """Isolation Forest 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 이상 점수 분포
        axes[0, 0].hist(train_scores, bins=50, alpha=0.7, label='Train')
        axes[0, 0].hist(test_scores, bins=50, alpha=0.7, label='Test')
        axes[0, 0].set_xlabel('이상 점수')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].set_title('Isolation Forest 이상 점수 분포')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Train 점수 시계열
        axes[0, 1].scatter(range(len(train_scores)), train_scores,
                          c=train_anomalies, cmap='coolwarm', alpha=0.6, s=1)
        axes[0, 1].set_xlabel('샘플 인덱스')
        axes[0, 1].set_ylabel('이상 점수')
        axes[0, 1].set_title('Train 이상 점수')
        axes[0, 1].grid(True)

        # Test 점수 시계열
        axes[1, 0].scatter(range(len(test_scores)), test_scores,
                          c=test_anomalies, cmap='coolwarm', alpha=0.6, s=1)
        axes[1, 0].set_xlabel('샘플 인덱스')
        axes[1, 0].set_ylabel('이상 점수')
        axes[1, 0].set_title('Test 이상 점수')
        axes[1, 0].grid(True)

        # 이상치 비율
        train_rate = np.mean(train_anomalies) * 100
        test_rate = np.mean(test_anomalies) * 100

        axes[1, 1].bar(['Train', 'Test'], [train_rate, test_rate], color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_ylabel('이상치 비율 (%)')
        axes[1, 1].set_title('이상치 탐지 비율')
        axes[1, 1].grid(True, axis='y')

        for i, v in enumerate([train_rate, test_rate]):
            axes[1, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('isolation_forest_anomaly_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_lstm_autoencoder_results(self, history, train_mse, test_mse, threshold,
                                       train_anomalies, test_anomalies):
        """LSTM Autoencoder 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 학습 히스토리
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('LSTM Autoencoder 학습 히스토리')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 재구성 오차 분포
        axes[0, 1].hist(train_mse, bins=50, alpha=0.7, label='Train')
        axes[0, 1].hist(test_mse, bins=50, alpha=0.7, label='Test')
        axes[0, 1].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        axes[0, 1].set_xlabel('재구성 오차 (MSE)')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('재구성 오차 분포')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Train 이상치
        axes[1, 0].scatter(range(len(train_mse)), train_mse, c=train_anomalies,
                          cmap='coolwarm', alpha=0.6, s=1)
        axes[1, 0].axhline(threshold, color='r', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('샘플 인덱스')
        axes[1, 0].set_ylabel('재구성 오차')
        axes[1, 0].set_title('Train 이상치 탐지')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Test 이상치
        axes[1, 1].scatter(range(len(test_mse)), test_mse, c=test_anomalies,
                          cmap='coolwarm', alpha=0.6, s=1)
        axes[1, 1].axhline(threshold, color='r', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('샘플 인덱스')
        axes[1, 1].set_ylabel('재구성 오차')
        axes[1, 1].set_title('Test 이상치 탐지')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('lstm_autoencoder_anomaly_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_models(self):
        """MLflow에서 모델 비교"""
        print("\n📊 MLflow에서 모델 성능 비교")

        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            print("   ⚠️ 실험을 찾을 수 없습니다.")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        print(f"\n{'Model Type':<20} {'Test Anomaly Rate':<20} {'Run ID':<40}")
        print("-" * 80)

        for run in runs:
            model_type = run.data.params.get('model_type', 'Unknown')
            test_anomaly_rate = run.data.metrics.get('test_anomaly_rate', 0)
            run_id = run.info.run_id

            print(f"{model_type:<20} {test_anomaly_rate:<20.2f} {run_id:<40}")

        print("\n💡 MLflow UI에서 더 자세한 비교를 확인하세요:")
        print("   mlflow ui --port 5000")


def main():
    """메인 실행 함수"""
    import argparse
    ap = argparse.ArgumentParser(description='CBM 이상 탐지 모델 학습 (MLflow 추적)')
    ap.add_argument('--data', default='preprocessed_shaft_data.parquet',
                    help='전처리된 특징 parquet 경로 '
                         '(실데이터 또는 evaluation/synthetic_data_generator.py 출력)')
    args = ap.parse_args()

    print("=" * 80)
    print("🔧 CBM 추진축계 이상 탐지 시스템 - MLflow 통합 버전")
    print("=" * 80)

    # 초기화
    detector = AnomalyDetectionMLflow(
        data_path=args.data,
        experiment_name='CBM_Shaft_Anomaly_Detection'
    )

    # 데이터 준비
    X_train, X_test = detector.prepare_data(test_size=0.2, random_state=42)

    # 1. Autoencoder 학습
    autoencoder, ae_threshold = detector.train_autoencoder(
        encoding_dim=32,
        epochs=50,
        batch_size=256
    )

    # 2. Isolation Forest 학습
    iso_forest = detector.train_isolation_forest(
        contamination=0.05,
        n_estimators=100
    )

    # 3. LSTM Autoencoder 학습
    lstm_ae, lstm_threshold = detector.train_lstm_autoencoder(
        encoding_dim=32,
        epochs=50,
        batch_size=256,
        timesteps=10
    )

    # 모델 비교
    detector.compare_models()

    print("\n" + "=" * 80)
    print("✅ 모든 모델 학습 및 MLflow 추적 완료!")
    print("=" * 80)
    print("\n📊 MLflow UI 실행:")
    print("   mlflow ui --port 5000")
    print("   브라우저에서 http://localhost:5000 접속")
    print("\n🔍 모델 로드 예제:")
    print(f"   import mlflow")
    print(f"   model = mlflow.keras.load_model('runs:/{detector.autoencoder_run_id}/model')")


if __name__ == "__main__":
    main()
