import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

DATASET_PATH = "augmented_dataset_500.csv"

# ── Load & clean dataset ──────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
df.columns = [c.lstrip('\ufeff') for c in df.columns]
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

print(f"\n  ⚠  Only {len(df)} samples found.")
print("     Cross-validation (5-fold) is used for reliable evaluation.")
print("     A small hold-out test set (20%) is also shown for reference.\n")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train samples : {len(X_train)}")
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

mse        = mean_squared_error(y_test, y_pred)
rmse       = np.sqrt(mse)
mae        = mean_absolute_error(y_test, y_pred)
r2         = r2_score(y_test, y_pred)

# ── Accuracy metrics (regression) ────────────────────────────────────────────
# Mean Absolute Percentage Error (MAPE) — % avg deviation per prediction
mape       = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Tolerance accuracy — % of predictions within ±T of actual value
tolerance  = 0.1   # ±0.1 units (adjust as needed)
within_tol = np.mean(np.abs(y_test - y_pred) <= tolerance) * 100

# Normalised RMSE — RMSE as % of the target range (easier to interpret)
y_range    = y.max() - y.min()
nrmse      = (rmse / y_range) * 100

print("\n" + "=" * 55)
print("       Hold-out Test Set Evaluation")
print("=" * 55)
print(f"  MSE                   : {mse:.6f}")
print(f"  RMSE                  : {rmse:.6f}")
print(f"  MAE                   : {mae:.6f}")
print(f"  R² Score              : {r2:.6f}   (1.0 = perfect fit)")
print(f"  MAPE                  : {mape:.2f}%  (avg % error per sample)")
print(f"  NRMSE                 : {nrmse:.2f}%  (RMSE as % of value range)")
print(f"  Tolerance accuracy    : {within_tol:.2f}%  (predictions within ±{tolerance})")
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
    print("\n  ⚠  WARNING: Negative mean CV R² indicates overfitting.")
    print("     The model has too few samples to generalise reliably.")
    print("     Recommendation: collect more data (ideally 200+ samples).")
print("=" * 55)

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/rf_model_wilkinson2port.pkl"
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
         'o-', markersize=5, linewidth=1.2, label='Measured data')
ax1.set_title("Raw Dataset: r2 vs Port 1")
ax1.set_xlabel("r2")
ax1.set_ylabel("Port 1")
ax1.legend()
ax1.grid(True, alpha=0.4)

# Plot 2 — Predicted vs Actual over test sample index
ax2 = axes[1]
sample_idx = np.arange(len(X_test))
ax2.plot(sample_idx, result_df['port1_actual'].values,
         'o-', label='Actual',    markersize=6, linewidth=1)
ax2.plot(sample_idx, result_df['port1_pred'].values,
         'x--', label='Predicted', markersize=6, linewidth=1)
ax2.set_title(f"Port 1 — Actual vs Predicted  (test n={len(X_test)})")
ax2.set_xlabel("Sample index")
ax2.set_ylabel("Port 1 value")
ax2.legend()
ax2.grid(True, alpha=0.4)

# Plot 3 — Scatter with y = x ideal line
ax3 = axes[2]
ax3.scatter(result_df['port1_actual'], result_df['port1_pred'],
            s=60, alpha=0.8, label='Predictions')
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

# Add summary text box to plot 2
summary = (f"R²: {r2:.4f}  |  RMSE: {rmse:.4f}\n"
           f"MAPE: {mape:.2f}%  |  Tol±{tolerance}: {within_tol:.1f}%")
axes[1].text(0.02, 0.04, summary, transform=axes[1].transAxes,
             fontsize=8.5, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("2-Port Wilkinson Power Combiner — Random Forest", fontsize=13)
plt.tight_layout()

PLOT_PATH = "plots/training_plot_wilkinson2port.png"
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"\n  Plot saved to '{PLOT_PATH}'")