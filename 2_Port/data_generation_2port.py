import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ── Original cleaned dataset (57 points) ─────────────────────────────────────
data = {
    'r2': [
        0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065,
        0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115,
        0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165,
        0.170, 0.175, 0.180, 0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215,
        0.220, 0.225, 0.230, 0.235, 0.240, 0.245, 0.250, 0.255, 0.260, 0.265,
        0.270, 0.275, 0.280, 0.285, 0.290, 0.295, 0.300
    ],
    'Port 1': [
        2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 1.9650, 2.0000, 1.9156,
        1.8000, 1.7500, 1.6400, 1.6600, 1.6600, 1.6990, 1.7000, 1.6982, 1.7112, 1.7400,
        1.7700, 1.7700, 1.8350, 1.9200, 1.8800, 1.9200, 2.0000, 2.1000, 1.8500, 0.5300,
        0.3923, 0.8700, 0.8770, 0.9200, 1.0500, 1.1146, 1.1500, 1.1821, 1.0000, 0.6700,
        1.1200, 1.7200, 1.8000, 1.7400, 1.7000, 1.7000, 1.6990, 1.6427, 1.5338, 1.3900,
        1.2684, 1.3000, 1.3678, 1.4000, 1.3500, 1.5335, 1.5500
    ]
}

df_base = pd.DataFrame(data)

# ── Step 1: Fit cubic spline on the 57 clean points ──────────────────────────
cs = CubicSpline(df_base['r2'].values, df_base['Port 1'].values)

# ── Step 2: Generate 500 evenly-spaced r2 values across the same range ───────
TARGET      = 1000
r2_new      = np.linspace(df_base['r2'].min(), df_base['r2'].max(), TARGET)
port1_spline = cs(r2_new)

# ── Step 3: Add realistic measurement noise ───────────────────────────────────
# Noise level calibrated to match the variation seen in original data (~0.5–1%)
np.random.seed(42)
noise_std   = 0.012                          # ≈ 0.7% of typical Port 1 value ~1.7
noise       = np.random.normal(0, noise_std, TARGET)
port1_noisy = port1_spline + noise

# Clip to physically valid range (Port 1 cannot exceed 2.1 or go below 0.3)
port1_final = np.clip(port1_noisy, 0.30, 2.10).round(4)
r2_final    = np.round(r2_new, 6)

# ── Step 4: Build DataFrame ───────────────────────────────────────────────────
df_aug = pd.DataFrame({
    'Sl_No'  : range(1, TARGET + 1),
    'r2'     : r2_final,
    'Port 1' : port1_final
})

# ── Step 5: Save ──────────────────────────────────────────────────────────────
OUT_PATH = "augmented_dataset_500.csv"
df_aug.to_csv(OUT_PATH, index=False)

print("=" * 55)
print("        AUGMENTED DATASET SUMMARY")
print("=" * 55)
print(f"  Total rows        : {len(df_aug)}")
print(f"  r2  range         : {df_aug['r2'].min():.4f}  →  {df_aug['r2'].max():.4f}")
print(f"  Port 1 range      : {df_aug['Port 1'].min():.4f}  →  {df_aug['Port 1'].max():.4f}")
print(f"  Port 1 mean       : {df_aug['Port 1'].mean():.4f}")
print(f"  Port 1 std dev    : {df_aug['Port 1'].std():.4f}")
print(f"  Saved to          : {OUT_PATH}")
print("=" * 55)
print(f"\nFirst 10 rows:")
print(df_aug.head(10).to_string(index=False))
