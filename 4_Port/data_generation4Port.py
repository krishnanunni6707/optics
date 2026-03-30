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
        2.5700, 2.4000, 2.5000, 2.6200, 2.6600, 2.5800, 2.3900, 2.3900, 1.4300, 1.5700,
        1.6100, 1.7800, 2.2200, 2.4300, 2.5660, 2.5890, 2.6000, 2.5867, 2.5742, 2.6100,
        2.6950, 2.7110, 2.7300, 2.7400, 2.7400, 2.8300, 3.0000, 2.8000, 2.7300, 2.7600,
        2.7500, 2.4500, 2.6500, 2.5000, 2.3000, 2.1250, 2.0000, 1.9635, 1.9700, 1.9300,
        1.8000, 1.8600, 1.8300, 1.8500, 2.3000, 2.5500, 2.6500, 2.2300, 1.8600, 1.7500,
        2.0000, 2.5600, 2.4500, 2.8800, 2.5600, 2.4300, 2.0000
    ]
}

df_base = pd.DataFrame(data)

# ── Step 1: Fit cubic spline on the 57 clean points ──────────────────────────
cs = CubicSpline(df_base['r2'].values, df_base['Port 1'].values)

# ── Step 2: Generate 500 evenly-spaced r2 values ─────────────────────────────
TARGET       = 500
r2_new       = np.linspace(df_base['r2'].min(), df_base['r2'].max(), TARGET)
port1_spline = cs(r2_new)

# ── Step 3: Add realistic measurement noise ───────────────────────────────────
# Noise std = 0.015 ≈ 0.6% of typical Port 1 value ~2.5
np.random.seed(42)
noise        = np.random.normal(0, 0.015, TARGET)
port1_noisy  = port1_spline + noise

# Clip to physically valid range (measured min=1.43, max=3.0)
port1_final  = np.clip(port1_noisy, 1.43, 3.00).round(4)
r2_final     = np.round(r2_new, 6)

# ── Step 4: Build DataFrame ───────────────────────────────────────────────────
df_aug = pd.DataFrame({
    'Sl_No'  : range(1, TARGET + 1),
    'r2'     : r2_final,
    'Port 1' : port1_final
})

# ── Step 5: Save ──────────────────────────────────────────────────────────────
OUT_PATH = "4_Port_Power_Combiner_Augmented_500.csv"
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