import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

DATASET_PATH = "4_Port_Power_Combiner_Augmented_500.csv"

# ── Load & clean dataset ──────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
df = df[['Sl_No', 'r2', 'Port 1']]
df.dropna(subset=['r2', 'Port 1'], inplace=True)
df.reset_index(drop=True, inplace=True)

print("=" * 55)
print("              DATASET INFO")
print("=" * 55)
print(f"  Total valid rows  : {len(df)}")
print(f"  r2  range         : {df['r2'].min():.4f}  →  {df['r2'].max():.4f}")
print(f"  Port 1 range      : {df['Port 1'].min():.4f}  →  {df['Port 1'].max():.4f}")
print("=" * 55)

# ── Features & target ─────────────────────────────────────────────────────────
X = df[['r2']].values
y = df['Port 1'].values

# ── Train / Test split (80 / 20) ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")

# ── Train model ───────────────────────────────────────────────────────────────
print("\n  Training Random Forest model …")
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  Training complete.")

# ── Hold-out test evaluation ──────────────────────────────────────────────────
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# ── Accuracy metrics ──────────────────────────────────────────────────────────
mape      = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
tolerance = 0.1
within_tol = np.mean(np.abs(y_test - y_pred) <= tolerance) * 100
y_range   = y.max() - y.min()
nrmse     = (rmse / y_range) * 100

print("\n" + "=" * 55)
print("       Hold-out Test Set Evaluation")
print("=" * 55)
print(f"  MSE                : {mse:.6f}")
print(f"  RMSE               : {rmse:.6f}")
print(f"  MAE                : {mae:.6f}")
print(f"  R² Score           : {r2:.6f}   (1.0 = perfect fit)")
print(f"  MAPE               : {mape:.2f}%  (avg % error per sample)")
print(f"  NRMSE              : {nrmse:.2f}%  (RMSE as % of value range)")
print(f"  Tolerance accuracy : {within_tol:.2f}%  (predictions within ±{tolerance})")
print("=" * 55)

# ── 5-Fold Cross-validation ───────────────────────────────────────────────────
cv_r2   = cross_val_score(model, X, y, cv=5, scoring='r2')
cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_squared_error'))

print("\n" + "=" * 55)
print("       5-Fold Cross-Validation (full dataset)")
print("=" * 55)
print(f"  R²   per fold : {np.round(cv_r2, 4)}")
print(f"  R²   mean     : {cv_r2.mean():.6f}  ±  {cv_r2.std():.6f}")
print(f"  RMSE per fold : {np.round(cv_rmse, 4)}")
print(f"  RMSE mean     : {cv_rmse.mean():.6f}  ±  {cv_rmse.std():.6f}")
if cv_r2.mean() < 0:
    print("\n  ⚠  WARNING: Negative mean CV R² — consider collecting more data.")
print("=" * 55)

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/rf_model_4port_combiner.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\n  Model saved to '{MODEL_PATH}'")

# ── Results DataFrame ─────────────────────────────────────────────────────────
result_df = pd.DataFrame({
    'r2'           : X_test.flatten(),
    'port1_actual' : y_test,
    'port1_pred'   : y_pred,
    'abs_error'    : np.abs(y_test - y_pred),
    'pct_error'    : np.abs((y_test - y_pred) / y_test) * 100
})

# --- Modified Print Section ---
num_show = 20  # Change this to 10 or 20 as needed
print(f"\nPredicted vs Actual — showing first {num_show} of {len(result_df)} test rows:")
print(result_df.head(num_show).to_string(index=False, float_format='%.4f'))

# ── Plots ─────────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1 — Raw dataset curve
ax1 = axes[0]
sort_idx = np.argsort(df['r2'].values)
ax1.plot(df['r2'].values[sort_idx], df['Port 1'].values[sort_idx],
         '.', markersize=3, alpha=0.7, color='steelblue', label='Augmented data (500 pts)')
ax1.set_title("Dataset: r2 vs Port 1")
ax1.set_xlabel("r2")
ax1.set_ylabel("Port 1")
ax1.legend()
ax1.grid(True, alpha=0.4)

# Plot 2 — Actual vs Predicted (sorted by r2)
ax2 = axes[1]
ax2.plot(result_df['r2'], result_df['port1_actual'],
         'o-', label='Actual',    markersize=5, linewidth=1.2)
ax2.plot(result_df['r2'], result_df['port1_pred'],
         'x--', label='Predicted', markersize=5, linewidth=1.2)
ax2.set_title(f"Port 1 — Actual vs Predicted  (test n={len(X_test)})")
ax2.set_xlabel("r2")
ax2.set_ylabel("Port 1 value")
ax2.legend()
ax2.grid(True, alpha=0.4)
summary = (f"R²: {r2:.4f}  |  RMSE: {rmse:.4f}\n"
           f"MAPE: {mape:.2f}%  |  Tol±{tolerance}: {within_tol:.1f}%")
ax2.text(0.02, 0.04, summary, transform=ax2.transAxes, fontsize=8.5,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3 — Scatter: perfect model lies on y = x
ax3 = axes[2]
ax3.scatter(result_df['port1_actual'], result_df['port1_pred'],
            s=40, alpha=0.7, label='Predictions')
lims = [
    min(result_df['port1_actual'].min(), result_df['port1_pred'].min()) - 0.05,
    max(result_df['port1_actual'].max(), result_df['port1_pred'].max()) + 0.05
]
ax3.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect fit (y = x)')
ax3.set_title(f"Actual vs Predicted — scatter  (R² = {r2:.4f})")
ax3.set_xlabel("Actual Port 1")
ax3.set_ylabel("Predicted Port 1")
ax3.legend()
ax3.grid(True, alpha=0.4)

plt.suptitle("4-Port Power Combiner — Random Forest Regression", fontsize=13)
plt.tight_layout()

PLOT_PATH = "plots/training_plot_4port_combiner.png"
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"\n  Plot saved to '{PLOT_PATH}'")

# ── Plot 4 — Full-range smooth prediction curve ───────────────────────────────
r2_range = np.linspace(df['r2'].min(), df['r2'].max(), 500).reshape(-1, 1)
y_curve  = model.predict(r2_range)

plt.figure(figsize=(10, 5))
plt.scatter(df['r2'], df['Port 1'], color='steelblue', zorder=5,
            label='All data points', s=10, alpha=0.5)
plt.plot(r2_range, y_curve, color='tomato', linewidth=2,
         label='RF prediction curve')
plt.title("4-Port Power Combiner — Random Forest Regression Curve")
plt.xlabel("r2")
plt.ylabel("Port 1 Value")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

CURVE_PATH = "plots/prediction_curve_4port_combiner.png"
plt.savefig(CURVE_PATH, dpi=150)
plt.show()
print(f"  Prediction curve saved to '{CURVE_PATH}'")
