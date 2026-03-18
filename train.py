import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATASET_PATH = "4port_dataset_500.csv"

# ______ Loaing dataset
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

print("=" * 50)
print("         DATASET INFO")
print("=" * 50)
print(f"  Total rows loaded : {len(df)}")
print(f"  Columns           : {list(df.columns)}")
print("=" * 50)

#  __________ feature Extraction
# here we extract the features for predict 
X = df[['r2']].values
y = df[['port1', 'port2', 'port3', 'port4']].values

# ______Train test splitting 
# here splits 80% data for training and 20% data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")

# _______Training the model

print("\n  Training Random Forest Model...")
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, random_state=42)
)
model.fit(X_train, y_train)
print("   Training complete.")

# _________Evaluating the model
y_pred = model.predict(X_test)

mse      = mean_squared_error(y_test, y_pred)
rmse     = np.sqrt(mse)
mae      = mean_absolute_error(y_test, y_pred)
r2       = r2_score(y_test, y_pred)
accuracy = r2 * 100
print("\n" + "=" * 50)
print("   Model Evaluation on Test Data")
print("=" * 50)
print(f"  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")
print(f"  MSE           : {mse:.6f}")
print(f"  RMSE          : {rmse:.6f}")
print(f"  MAE           : {mae:.6f}")
print(f"  R² Score      : {r2:.6f}")
print(f"  Accuracy      : {accuracy:.2f}%")
print("=" * 50)

# ______Saving the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rf_model.pkl")
print("\n   Model saved to 'model/rf_model.pkl'")

# Ploting the results
result_df = pd.DataFrame(
    y_pred,
    columns=['port1_pred', 'port2_pred', 'port3_pred', 'port4_pred']
)
result_df[['port1_actual', 'port2_actual', 'port3_actual', 'port4_actual']] = y_test
result_df['r2'] = X_test.flatten()

print(f"\nPredicted vs Actual Test Set (first 10 of {len(result_df)} rows):")
print(result_df[['r2',
                  'port1_actual', 'port1_pred',
                  'port2_actual', 'port2_pred',
                  'port3_actual', 'port3_pred',
                  'port4_actual', 'port4_pred']].head(10).to_string(index=False))
ports      = ['port1', 'port2', 'port3', 'port4']
sample_idx = np.arange(len(X_test))
plt.figure(figsize=(16, 10))
for i, port in enumerate(ports, 1):
    plt.subplot(2, 2, i)
    actual    = result_df[f'{port}_actual'].values
    predicted = result_df[f'{port}_pred'].values
    plt.plot(sample_idx, actual,    'o-',  label='Actual',    markersize=2, linewidth=1)
    plt.plot(sample_idx, predicted, 'x--', label='Predicted', markersize=2, linewidth=1)
    plt.title(f"{port} Actual vs Predicted  (test n={len(X_test)})")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.4)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/training_plot.png", dpi=150)
plt.show()
print("\n   Training plot saved to 'plots/training_plot.png'")