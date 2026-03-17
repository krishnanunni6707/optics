import numpy as np
import pandas as pd

np.random.seed(42)
original = {
    'r2':    [0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,
              0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29],
    'input': [0.60,0.68,0.84,0.925,1.15,1.18,1.08,1.10,1.10,1.00,
              0.98,0.92,0.90,0.84,0.80,0.83,0.95,1.04,1.12,1.12],
    'port1': [0.180,0.140,0.150,0.128,0.640,0.054,0.215,0.192,0.180,0.190,
              0.218,0.210,0.220,0.260,0.280,0.230,0.160,0.320,0.310,0.240],
    'port2': [0.160,0.140,0.150,0.110,0.092,0.054,0.220,0.175,0.220,0.260,
              0.260,0.260,0.260,0.240,0.240,0.245,0.260,0.300,0.260,0.180],
    'port3': [0.068,0.100,0.100,0.0825,0.089,0.072,0.160,0.175,0.150,0.190,
              0.185,0.180,0.220,0.240,0.275,0.240,0.260,0.310,0.270,1.120],
    'port4': [0.074,0.100,0.950,0.070,0.100,0.110,0.140,0.175,0.220,0.250,
              0.260,0.260,0.250,0.225,0.210,0.260,0.160,0.220,0.280,1.130]
}

df_orig = pd.DataFrame(original)
from scipy.interpolate import CubicSpline

cs_input = CubicSpline(df_orig['r2'], df_orig['input'])
cs_port1 = CubicSpline(df_orig['r2'], df_orig['port1'])
cs_port2 = CubicSpline(df_orig['r2'], df_orig['port2'])
cs_port3 = CubicSpline(df_orig['r2'], df_orig['port3'])
cs_port4 = CubicSpline(df_orig['r2'], df_orig['port4'])

n = 500
r2_vals = np.random.uniform(0.10, 0.29, n)
input_base = cs_input(r2_vals)
port1_base = cs_port1(r2_vals)
port2_base = cs_port2(r2_vals)
port3_base = cs_port3(r2_vals)
port4_base = cs_port4(r2_vals)
noise_scale = 0.01
input_vals = input_base + np.random.normal(0, noise_scale * 2,  n)
port1_vals = port1_base + np.random.normal(0, noise_scale,      n)
port2_vals = port2_base + np.random.normal(0, noise_scale,      n)
port3_vals = port3_base + np.random.normal(0, noise_scale,      n)
port4_vals = port4_base + np.random.normal(0, noise_scale,      n)

df_gen = pd.DataFrame({
    'r2'   : np.round(r2_vals,   6),
    'input': np.round(input_vals,6),
    'port1': np.round(port1_vals,6),
    'port2': np.round(port2_vals,6),
    'port3': np.round(port3_vals,6),
    'port4': np.round(port4_vals,6),
})
df_gen['input'] = df_gen['input'].clip(0.50, 1.25)
df_gen['port1'] = df_gen['port1'].clip(0.00, 0.70)
df_gen['port2'] = df_gen['port2'].clip(0.00, 0.35)
df_gen['port3'] = df_gen['port3'].clip(0.00, 1.20)
df_gen['port4'] = df_gen['port4'].clip(0.00, 1.20)

df_gen.to_csv("4port_dataset_500.csv", index=False)
print(f"Generated {len(df_gen)} rows saved to '4port_dataset_500.csv'")
print(df_gen.head(10).to_string(index=False))