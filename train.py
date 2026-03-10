import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATASET_PATH = "4port_dataset_1000.csv"
# ----------- Load CSV Dataset -----------
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

print("=" * 50)
print("         DATASET INFO")
print("=" * 50)
print(f"  Total rows loaded : {len(df)}")
print(f"  Columns           : {list(df.columns)}")
print("=" * 50)

# ----------- Features and Targets -----------
X = df[['r2']].values
y = df[['port1', 'port2', 'port3', 'port4']].values

# ----------- Train Random Forest -----------
print("\n  Training Random Forest Model...")

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)

model.fit(X, y)
print("   Training complete.")

# ----------- Make Predictions on Training Data -----------
y_pred = model.predict(X)

# ----------- Evaluation Metrics -----------
mse      = mean_squared_error(y, y_pred)
rmse     = np.sqrt(mse)
mae      = mean_absolute_error(y, y_pred)
r2       = r2_score(y, y_pred)
accuracy = r2 * 100

print("\n" + "=" * 50)
print("       Model Evaluation on Training Data")
print("=" * 50)
print(f"  Total samples : {len(df)}")
print(f"  MSE           : {mse:.6f}")
print(f"  RMSE          : {rmse:.6f}")
print(f"  MAE           : {mae:.6f}")
print(f"  R² Score      : {r2:.6f}")
print(f"  Accuracy      : {accuracy:.2f}%")
print("=" * 50)

# ----------- Save Model -----------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rf_model.pkl")
print("\n   Model saved to 'model/rf_model.pkl'")

# ----------- Predicted vs Actual Table -----------
result_df = pd.DataFrame(
    y_pred,
    columns=['port1_pred', 'port2_pred', 'port3_pred', 'port4_pred']
)
result_df[['port1_actual', 'port2_actual', 'port3_actual', 'port4_actual']] = \
    df[['port1', 'port2', 'port3', 'port4']].values
result_df = pd.concat([df[['r2']].reset_index(drop=True), result_df], axis=1)

print(f"\nPredicted vs Actual Table (first 10 of {len(result_df)} rows):")
print(result_df.head(10).to_string(index=False))

# ----------- Plot Predicted vs Actual -----------
ports      = ['port1', 'port2', 'port3', 'port4']
sample_idx = np.arange(len(df))

plt.figure(figsize=(16, 10))

for i, port in enumerate(ports, 1):
    plt.subplot(2, 2, i)

    actual    = df[port].values
    predicted = result_df[f'{port}_pred'].values

    plt.plot(sample_idx, actual,    'o-',  label='Actual',    markersize=2, linewidth=1)
    plt.plot(sample_idx, predicted, 'x--', label='Predicted', markersize=2, linewidth=1)

    plt.title(f"{port} → Actual vs Predicted  (n={len(df)})")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("model/training_plot.png", dpi=150)
plt.show()
print("\n   Training plot saved to 'model/training_plot.png'")
print("  Run 'predict.py' to make predictions using the saved model.")