import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ----------- Original 20-row dataset -----------
original = {
    'r2':    [0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
              0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29],
    'port1': [0.237,0.185,0.208,0.160,0.110,0.120,0.360,0.220,0.213,0.257,
              0.272,0.261,0.281,0.321,0.379,0.307,0.209,0.476,0.441,0.315],
    'port2': [0.224,0.196,0.168,0.150,0.140,0.180,0.300,0.190,0.284,0.263,
              0.322,0.320,0.317,0.308,0.3117,0.305,0.436,0.437,0.400,0.236],
    'port3': [0.100,0.138,0.132,0.120,0.150,0.160,0.206,0.220,0.210,0.255,
              0.222,0.247,0.276,0.331,0.365,0.303,0.260,0.457,0.376,0.242],
    'port4': [0.097,0.137,0.141,0.110,0.142,0.150,0.220,0.240,0.290,0.241,
              0.323,0.334,0.314,0.286,0.274,0.295,0.246,0.270,0.369,0.224],
}

df_orig = pd.DataFrame(original)

# ----------- Fit cubic spline on original 20 rows -----------
cs_port1 = CubicSpline(df_orig['r2'], df_orig['port1'])
cs_port2 = CubicSpline(df_orig['r2'], df_orig['port2'])
cs_port3 = CubicSpline(df_orig['r2'], df_orig['port3'])
cs_port4 = CubicSpline(df_orig['r2'], df_orig['port4'])

# ----------- Generate 500 rows -----------
np.random.seed(42)
n       = 500
r2_vals = np.random.uniform(0.10, 0.29, n)
noise   = 0.008

port1_vals = cs_port1(r2_vals) + np.random.normal(0, noise, n)
port2_vals = cs_port2(r2_vals) + np.random.normal(0, noise, n)
port3_vals = cs_port3(r2_vals) + np.random.normal(0, noise, n)
port4_vals = cs_port4(r2_vals) + np.random.normal(0, noise, n)

df_gen = pd.DataFrame({
    'r2'   : np.round(r2_vals,             6),
    'port1': np.round(port1_vals.clip(0, 0.55), 6),
    'port2': np.round(port2_vals.clip(0, 0.50), 6),
    'port3': np.round(port3_vals.clip(0, 0.50), 6),
    'port4': np.round(port4_vals.clip(0, 0.45), 6),
})

df_gen.to_csv("4port_dataset_500.csv", index=False)

print("=" * 50)
print("      Dataset Generation Complete")
print("=" * 50)
print(f"  Rows generated : {len(df_gen)}")
print(f"  r2 range       : {df_gen['r2'].min():.4f} – {df_gen['r2'].max():.4f}")
print(f"  Saved to       : 4port_dataset_500.csv")
print("=" * 50)
print("\nFirst 5 rows:")
print(df_gen.head().to_string(index=False))