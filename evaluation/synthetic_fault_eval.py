"""
Labeled evaluation of the three CBM models on the synthetic fault benchmark.

The real shaft data is unlabeled, so the original project could only report
"X% of windows flagged" — which says nothing about whether the right windows
were flagged. This script injects four physically-motivated fault types with
known labels (see synthetic_data_generator.py) and measures actual detection
quality: PR-AUC, precision/recall/F1, and per-fault recall for
  - Isolation Forest        (point-wise, tree ensemble)
  - Dense Autoencoder       (point-wise, reconstruction error)
  - LSTM Autoencoder        (sequence, reconstruction error, timesteps=10)

Thresholds are calibrated on training (healthy) scores at the 95th percentile —
the same rule the production pipeline uses — NOT tuned on test labels.

Run:  python evaluation/synthetic_fault_eval.py
Outputs docs/analysis/synthetic_fault_eval.png + printed metric table.
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic_data_generator import generate_benchmark, FAULT_TYPES

SEED = 42
TIMESTEPS = 10
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- data ----------
print('Generating synthetic benchmark...')
train_df, test_df = generate_benchmark(seed=SEED)
feature_cols = [c for c in train_df.columns if c != 'timestamp']
print(f'  train (healthy): {train_df.shape}, test: {test_df.shape}, '
      f'fault windows: {int(test_df["label"].sum())}')

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
y_test = test_df['label'].values
fault_type = test_df['fault_type'].values

results = {}

def evaluate(name, train_scores, test_scores):
    """Higher score = more anomalous. Threshold = train 95th percentile."""
    thr = np.percentile(train_scores, 95)
    pred = test_scores > thr
    tp = int(np.sum(pred & (y_test == 1)))
    fp = int(np.sum(pred & (y_test == 0)))
    fn = int(np.sum(~pred & (y_test == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ap = average_precision_score(y_test, test_scores)
    per_fault = {ft: float(np.mean(pred[fault_type == ft])) for ft in FAULT_TYPES}
    fp_rate = float(np.mean(pred[y_test == 0]))
    results[name] = dict(ap=ap, prec=prec, rec=rec, f1=f1, fp=fp, fp_rate=fp_rate,
                         per_fault=per_fault, scores=test_scores, thr=thr, pred=pred)
    print(f'{name:<18} PR-AUC {ap:.3f} | P {prec:.2f} R {rec:.2f} F1 {f1:.2f} '
          f'(FP {fp}, {fp_rate:.1%} of normal) | ' +
          ' '.join(f'{ft[:6]}:{v:.0%}' for ft, v in per_fault.items()))

# ---------- 1. Isolation Forest ----------
print('\nTraining Isolation Forest...')
iso = IsolationForest(contamination=0.05, n_estimators=100, random_state=SEED, n_jobs=-1)
iso.fit(X_train)
evaluate('IsolationForest', -iso.score_samples(X_train), -iso.score_samples(X_test))

# ---------- 2. Dense Autoencoder ----------
print('Training Dense Autoencoder...')
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.utils.set_random_seed(SEED)

inp = Input(shape=(X_train.shape[1],))
enc = Dense(32, activation='relu')(inp)
enc = Dense(16, activation='relu')(enc)
dec = Dense(32, activation='relu')(enc)
dec = Dense(X_train.shape[1], activation='linear')(dec)
ae = Model(inp, dec)
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1,
       callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
       verbose=0)

ae_train = np.mean((X_train - ae.predict(X_train, verbose=0)) ** 2, axis=1)
ae_test = np.mean((X_test - ae.predict(X_test, verbose=0)) ** 2, axis=1)
evaluate('Autoencoder', ae_train, ae_test)

# ---------- 3. LSTM Autoencoder ----------
# Sequences must not cross voyage boundaries; label = last window's label.
print('Training LSTM Autoencoder...')

def to_sequences(X, boundaries):
    seqs, idx = [], []
    start = 0
    for end in boundaries:
        for i in range(start, end - TIMESTEPS + 1):
            seqs.append(X[i:i + TIMESTEPS])
            idx.append(i + TIMESTEPS - 1)   # sequence labeled by its last window
        start = end
    return np.array(seqs), np.array(idx)

train_bounds = [len(train_df)]
test_bounds = np.where(np.diff(test_df['timestamp'].values) < 0)[0] + 1
test_bounds = list(test_bounds) + [len(test_df)]

S_train, _ = to_sequences(X_train, train_bounds)
S_test, seq_idx = to_sequences(X_test, test_bounds)
y_seq = y_test[seq_idx]
fault_seq = fault_type[seq_idx]

inp = Input(shape=(TIMESTEPS, X_train.shape[1]))
enc = LSTM(32, activation='tanh')(inp)
dec = RepeatVector(TIMESTEPS)(enc)
dec = LSTM(32, activation='tanh', return_sequences=True)(dec)
dec = TimeDistributed(Dense(X_train.shape[1]))(dec)
lstm_ae = Model(inp, dec)
lstm_ae.compile(optimizer='adam', loss='mse')
lstm_ae.fit(S_train, S_train, epochs=50, batch_size=64, validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=0)

lstm_train = np.mean((S_train - lstm_ae.predict(S_train, verbose=0)) ** 2, axis=(1, 2))
lstm_test = np.mean((S_test - lstm_ae.predict(S_test, verbose=0)) ** 2, axis=(1, 2))

# evaluate() uses window-level y_test/fault_type; swap in sequence-level views
y_test_full, fault_full = y_test, fault_type
y_test, fault_type = y_seq, fault_seq
evaluate('LSTM-AE (t=10)', lstm_train, lstm_test)
y_test, fault_type = y_test_full, fault_full

# ---------- figure ----------
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))

# (1) example fault voyage: IF score timeline with fault segment shaded
first_bearing = np.where(test_df['fault_type'] == 'bearing_wear')[0]
voyage_start = max(0, first_bearing[0] - 60)
voyage_end = min(len(test_df), first_bearing[-1] + 60)
seg = slice(voyage_start, voyage_end)
if_r = results['IsolationForest']
x_ax = np.arange(voyage_end - voyage_start)
axs[0].plot(x_ax, if_r['scores'][seg], color='royalblue', lw=0.9, label='IF anomaly score')
axs[0].axhline(if_r['thr'], color='r', ls='--', lw=1, label='threshold (train 95%)')
in_fault = (test_df['label'].values[seg] == 1)
axs[0].fill_between(x_ax, *axs[0].get_ylim(), where=in_fault,
                    color='crimson', alpha=0.12, label='bearing-wear segment (ramped)')
axs[0].set_xlabel('window (1 s)'); axs[0].set_ylabel('anomaly score')
axs[0].set_title('Progressive bearing-wear voyage: score vs injected fault')
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

# (2) per-fault recall by model
models = list(results)
x = np.arange(len(FAULT_TYPES)); w = 0.25
colors = ['royalblue', 'darkorange', 'seagreen']
for i, m in enumerate(models):
    axs[1].bar(x + (i - 1) * w, [results[m]['per_fault'][ft] for ft in FAULT_TYPES],
               w, color=colors[i], label=f"{m} (F1={results[m]['f1']:.2f})")
axs[1].set_xticks(x); axs[1].set_xticklabels(FAULT_TYPES, rotation=12)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel('Recall (calibrated threshold)')
axs[1].set_title('Per-fault recall by model')
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3, axis='y')

# (3) overall metrics
metrics = ['ap', 'prec', 'rec', 'f1']
labels_m = ['PR-AUC', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
for i, m in enumerate(models):
    axs[2].bar(x + (i - 1) * w, [results[m][k] for k in metrics], w,
               color=colors[i], label=m)
axs[2].set_xticks(x); axs[2].set_xticklabels(labels_m)
axs[2].set_ylim(0, 1.05); axs[2].set_title('Overall detection metrics')
axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'synthetic_fault_eval.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f'\nSaved: {out_path}')
print('EVAL_OK')
