<p align="center">
  <a href="README.md">English</a> | <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Ship Shaft CBM Anomaly Detection</h1>

<p align="center">Condition-based maintenance for a ship propulsion shaft — three unsupervised detectors tracked in MLflow, a Streamlit monitoring dashboard, and a labeled synthetic-fault benchmark (best F1 0.93)</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Benchmark-F1%200.93-green.svg" alt="F1 0.93">
</p>

Vibration and RPM sensors on a ship's propulsion shaft produce a continuous stream of
unlabeled data. This project learns the healthy signature of that stream with three
unsupervised models — **Isolation Forest**, a **dense autoencoder**, and an
**LSTM autoencoder** — tracks every training run in MLflow, and serves the detector
through a Streamlit operations dashboard with anomaly-cause attribution
(which sensor deviated, by how many σ).

Because the real data is unlabeled, "X% of windows flagged" says nothing about whether
the *right* windows were flagged. The repository therefore includes a
**physics-informed synthetic fault benchmark**: four classic rotating-machinery faults
are injected into simulated shaft vibration with known labels, and all three models are
scored on precision/recall/F1 — thresholds calibrated on healthy data only, never tuned
on test labels.

## Data

The original sensors are proprietary (propulsion-shaft monitoring of the Korea Maritime
& Ocean University training ship *Hanbada*): a tachometer (`RTTN_SPDMTR`) and four
vibration channels (shaft lower/upper, vertical, horizontal), summarized per 1-second
window as mean / std / RMS / peak-to-peak / kurtosis / skewness.

The raw data cannot be redistributed, so `evaluation/synthetic_data_generator.py`
produces a drop-in substitute with the same schema: 1×/2×/3× RPM harmonic composition,
centrifugal amplitude scaling, and sensor noise for the healthy state, plus four
injectable faults with per-channel sensitivity profiles:

| Fault | Physical signature | Feature response |
|---|---|---|
| Unbalance | 1×-RPM amplitude rise, radial-dominant | RMS / p2p ↑ |
| Misalignment | 2× (and 3×) harmonics, axial/horizontal | RMS ↑, waveform shape change |
| Bearing wear | Periodic impulses at the defect frequency | Kurtosis ↑↑ |
| Mechanical looseness | 0.5× sub-harmonic + broadband bursts | Std / skewness ↑ |

Every fault segment **ramps severity from 0.15 to 1.0** (progressive degradation), so
early low-severity windows are genuinely hard — the benchmark rewards early detection
rather than only fully-developed failures.

## Benchmark results

2,000 healthy training windows; 2,900 test windows of which 1,440 are faults
(4 types × 3 voyages each). Threshold = 95th percentile of training scores.
Run `python evaluation/synthetic_fault_eval.py` to reproduce:

| Model | PR-AUC | Precision | Recall | F1 | Unbalance | Misalign | Bearing | Looseness |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.845 | 0.91 | 0.75 | 0.82 | 89% | 90% | 41% | 78% |
| **Dense Autoencoder** | **0.975** | **0.96** | **0.91** | **0.93** | **100%** | **100%** | **72%** | **91%** |
| LSTM-AE (t=10) | 0.846 | 0.88 | 0.73 | 0.80 | 87% | 95% | 39% | 72% |

![Synthetic fault benchmark](docs/analysis/synthetic_fault_eval.png)

**What the labels reveal:**

- **The dense autoencoder wins decisively** (F1 0.93) — it reconstructs the joint
  healthy feature distribution and reacts to any deviation, catching unbalance and
  misalignment perfectly.
- **Bearing wear is the shared bottleneck** (39–72% recall): its signature lives almost
  entirely in kurtosis, and at low ramp severity the impulses drown in noise. This is
  the honest gap a "5% of windows flagged" report would never expose — and motivates
  envelope-spectrum features as the next step.
- **The LSTM-AE does not earn its cost here**: with slowly-varying cruise RPM, the
  temporal context adds noise rather than signal (it also has the highest false-alarm
  rate, 10.9%). Sequence models need faults with temporal structure — e.g. transient
  events — to justify themselves over point-wise models.

## Repository layout

| Path | Contents |
|---|---|
| `training/train_models_mlflow.py` | Trains all three models, logs params/metrics/artifacts/models to MLflow |
| `training/anomaly_cause_analysis.py` | Loads the latest run and attributes each anomaly to its top-deviating sensors |
| `evaluation/synthetic_data_generator.py` | Physics-informed shaft-vibration simulator (healthy + 4 fault types, labeled) |
| `evaluation/synthetic_fault_eval.py` | Labeled benchmark: PR-AUC / P / R / F1 per model, per-fault recall |
| `app.py`, `pages/` | Streamlit dashboard: real-time detection, batch analysis, statistics/trends, settings |
| `utils/model_utils.py` | Model loading (latest MLflow run auto-resolved), prediction, cause attribution |
| `docs/` | Model selection guide, dashboard guide, visualization guide, AWS deployment notes |

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate the synthetic data substitute (writes preprocessed_shaft_data.parquet)
python evaluation/synthetic_data_generator.py

# 2. Train all three models with MLflow tracking
python training/train_models_mlflow.py
mlflow ui --port 5000          # inspect runs at http://localhost:5000

# 3. Launch the monitoring dashboard (uses the latest Isolation Forest run)
streamlit run app.py

# 4. Reproduce the labeled benchmark (standalone — no MLflow needed)
python evaluation/synthetic_fault_eval.py
```

## Dashboard

Four pages built for an operations context: **Real-time Detection** (window-by-window
scoring with per-sensor cause attribution), **Batch Analysis** (upload CSV/parquet,
score in bulk), **Statistics & Trends** (anomaly-rate history), and **Settings**
(model info, threshold tuning). A `Dockerfile` is included for containerized
deployment; see [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md).

## Limitations

- The benchmark is synthetic: fault signatures follow textbook rotating-machinery
  theory, but real shaft data adds sea-state loading, hull-transmitted vibration, and
  sensor drift that the simulator does not model. Numbers are for **comparing models
  under identical conditions**, not absolute field performance.
- The real-data pipeline remains unsupervised; deploying a model chosen on the
  synthetic benchmark still requires field validation against maintenance logs.

## Related projects

- [Vehicle-Anomaly-Algorithm](https://github.com/chaeminyoon/Vehicle-Anomaly-Algorithm) — road-CCTV trajectory anomaly detection (same evaluate-honestly methodology)
- [AIS-Traffic-Model](https://github.com/chaeminyoon/AIS-Traffic-Model) / [AIS-Traffic-Ops](https://github.com/chaeminyoon/AIS-Traffic-Ops) — maritime traffic forecasting research & MLOps
