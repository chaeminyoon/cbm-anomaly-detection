"""
Synthetic shaft-vibration benchmark generator.

The real training data (KMOU training ship "Hanbada" propulsion-shaft sensors)
is proprietary and cannot be redistributed. This module generates a
physics-informed substitute so the full pipeline is reproducible AND
quantitatively evaluatable — faults are injected with known labels,
which the unlabeled real data never had.

Signal model (standard rotating-machinery diagnostics):
  healthy   : 1x-RPM sinusoid + weak 2x/3x harmonics + Gaussian noise,
              amplitude scales with RPM^2 (centrifugal loading)
  unbalance : 1x-RPM amplitude rises (radial channels strongest)
  misalign  : 2x (and 3x) harmonics rise (axial/horizontal strongest)
  bearing   : periodic impulses at the bearing defect frequency -> kurtosis rises
  looseness : intermittent broadband noise bursts + sub-harmonic (0.5x)

Each fault segment ramps severity 0 -> 1 (progressive degradation), so early
windows are genuinely hard — detection rates below 100% are expected and honest.

Output: one row per 1-second window with per-channel statistical features
(mean/std/rms/p2p/kurtosis/skewness) matching the real pipeline's feature style.
"""
import numpy as np
import pandas as pd
from scipy import stats

FS = 2048          # waveform sample rate (Hz) per 1-second window
CHANNELS = ['SH_LD_EBY_VIBRAT',   # shaft lower axial
            'SH_TD_EBY_VIBRAT',   # shaft upper axial
            'VD_EBY_VIBRAT',      # vertical radial
            'HD_EBY_VIBRAT']      # horizontal radial
FAULT_TYPES = ['unbalance', 'misalignment', 'bearing_wear', 'looseness']

# channel sensitivity per fault: (axial_lower, axial_upper, vertical, horizontal)
FAULT_CHANNEL_GAIN = {
    'unbalance':    np.array([0.3, 0.3, 1.0, 1.0]),   # radial-dominant
    'misalignment': np.array([1.0, 0.8, 0.4, 0.9]),   # axial/horizontal-dominant
    'bearing_wear': np.array([0.8, 0.6, 1.0, 0.7]),
    'looseness':    np.array([0.7, 0.7, 1.0, 1.0]),
}
BEARING_DEFECT_ORDER = 4.7   # defect frequency as multiple of shaft speed (BPFO-like)


def _window_waveforms(rpm, severity, fault, rng):
    """Generate one 1-second window of 4-channel vibration at given rpm."""
    t = np.arange(FS) / FS
    f_rot = rpm / 60.0
    base_amp = 0.5 + 1.5 * (rpm / 150.0) ** 2          # centrifugal scaling
    phase = rng.uniform(0, 2 * np.pi, size=3)

    # healthy composition per channel (slightly different mix per channel)
    ch_mix = np.array([[1.0, 0.15, 0.05],
                       [0.9, 0.20, 0.05],
                       [1.1, 0.10, 0.08],
                       [1.0, 0.12, 0.06]])
    sig = np.zeros((4, FS))
    for c in range(4):
        for h, (amp, ph) in enumerate(zip(ch_mix[c], phase)):
            sig[c] += base_amp * amp * np.sin(2 * np.pi * (h + 1) * f_rot * t + ph)
        sig[c] += rng.normal(0, 0.25 * base_amp, FS)   # sensor/flow noise

    if fault is None or severity <= 0:
        return sig

    g = FAULT_CHANNEL_GAIN[fault] * severity
    if fault == 'unbalance':
        extra = np.sin(2 * np.pi * f_rot * t + phase[0])
        sig += 1.8 * base_amp * g[:, None] * extra[None, :]
    elif fault == 'misalignment':
        extra2 = np.sin(2 * np.pi * 2 * f_rot * t + phase[1])
        extra3 = np.sin(2 * np.pi * 3 * f_rot * t + phase[2])
        sig += base_amp * g[:, None] * (1.6 * extra2 + 0.7 * extra3)[None, :]
    elif fault == 'bearing_wear':
        # repetitive impulses at defect frequency, exponentially decaying rings
        f_defect = BEARING_DEFECT_ORDER * f_rot
        impulse_times = np.arange(0, 1.0, 1.0 / max(f_defect, 1.0))
        impulse_times += rng.normal(0, 0.001, len(impulse_times))  # jitter
        ring = np.zeros(FS)
        for t0 in impulse_times:
            idx = int(t0 * FS)
            if 0 <= idx < FS - 64:
                dt = np.arange(64) / FS
                ring[idx:idx + 64] += np.exp(-dt / 0.004) * np.sin(2 * np.pi * 800 * dt)
        sig += 3.0 * base_amp * g[:, None] * ring[None, :]
    elif fault == 'looseness':
        sub = np.sin(2 * np.pi * 0.5 * f_rot * t + phase[0])       # 0.5x sub-harmonic
        burst_mask = (rng.random(FS) < 0.15).astype(float)
        burst = burst_mask * rng.normal(0, 1.0, FS)
        sig += base_amp * g[:, None] * (0.9 * sub + 1.2 * burst)[None, :]
    return sig


def _features(sig, rpm):
    """Per-channel statistical features for one window (matches real pipeline)."""
    row = {'RTTN_SPDMTR': rpm}
    for c, name in enumerate(CHANNELS):
        x = sig[c]
        row[f'{name}_mean'] = x.mean()
        row[f'{name}_std'] = x.std()
        row[f'{name}_rms'] = np.sqrt(np.mean(x ** 2))
        row[f'{name}_p2p'] = x.max() - x.min()
        row[f'{name}_kurtosis'] = stats.kurtosis(x)
        row[f'{name}_skewness'] = stats.skew(x)
    return row


def _rpm_profile(n, rng, lo=90, hi=150):
    """Slowly varying cruise RPM profile (random walk, clipped)."""
    steps = rng.normal(0, 0.8, n)
    rpm = np.clip(np.cumsum(steps) + rng.uniform(lo + 20, hi - 20), lo, hi)
    return rpm


def generate_voyage(n_windows, fault=None, fault_start=None, fault_len=None, seed=0):
    """
    Generate a contiguous voyage of n_windows 1-second windows.
    If fault is set, severity ramps 0->1 over [fault_start, fault_start+fault_len).
    Returns (features DataFrame, labels array, severity array).
    """
    rng = np.random.RandomState(seed)
    rpm = _rpm_profile(n_windows, rng)
    labels = np.zeros(n_windows, dtype=int)
    severity = np.zeros(n_windows)
    if fault is not None:
        end = min(fault_start + fault_len, n_windows)
        severity[fault_start:end] = np.linspace(0.15, 1.0, end - fault_start)
        labels[fault_start:end] = 1

    rows = []
    for i in range(n_windows):
        f = fault if labels[i] else None
        sig = _window_waveforms(rpm[i], severity[i], f, rng)
        rows.append(_features(sig, rpm[i]))
    df = pd.DataFrame(rows)
    df.insert(0, 'timestamp', np.arange(n_windows, dtype=float))
    return df, labels, severity


def generate_benchmark(seed=42, n_train=2000, n_test_normal=500, n_fault_voyages=3,
                       fault_voyage_len=200, fault_len=120):
    """
    Full benchmark: healthy training voyage + healthy test voyage
    + n_fault_voyages voyages per fault type, each with one ramped fault segment.
    Returns (train_df, test_df) where test_df has 'label' and 'fault_type' columns.
    """
    train_df, _, _ = generate_voyage(n_train, seed=seed)

    test_parts = []
    normal_df, labels, _ = generate_voyage(n_test_normal, seed=seed + 1)
    normal_df['label'] = labels
    normal_df['fault_type'] = 'normal'
    test_parts.append(normal_df)

    for fi, fault in enumerate(FAULT_TYPES):
        for v in range(n_fault_voyages):
            df, labels, sev = generate_voyage(
                fault_voyage_len, fault=fault,
                fault_start=(fault_voyage_len - fault_len) // 2,
                fault_len=fault_len, seed=seed + 100 + fi * 10 + v)
            df['label'] = labels
            df['fault_type'] = np.where(labels == 1, fault, 'normal')
            test_parts.append(df)

    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Generate synthetic shaft benchmark data')
    ap.add_argument('--out', default='preprocessed_shaft_data.parquet',
                    help='output parquet path (training features, healthy only)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train_df, test_df = generate_benchmark(seed=args.seed)
    train_df.to_parquet(args.out)
    test_out = args.out.replace('.parquet', '_test_labeled.parquet')
    test_df.to_parquet(test_out)
    print(f'train (healthy): {train_df.shape} -> {args.out}')
    print(f'test (labeled) : {test_df.shape} -> {test_out}')
    print(f'fault windows  : {int(test_df["label"].sum())} / {len(test_df)}')
