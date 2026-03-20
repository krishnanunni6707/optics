import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ___________Loading the models 
# here dataset is loaded for ploting its not actually 
# required for prediction purpose 
# just to plot the predicted data in the graph and checking!!!

MODEL_PATH   = "model/rf_model.pkl"
DATASET_PATH = "4port_dataset_500.csv"   

if not os.path.exists(MODEL_PATH):
    print("=" * 50)
    print("   Model not found!")
    print("  Please run 'train.py' first.")
    print("=" * 50)
    exit()
model = joblib.load(MODEL_PATH)
print("=" * 50)
print("   Model loaded from 'model/rf_model.pkl'")
print("=" * 50)
PORTS = ['port1', 'port2', 'port3', 'port4']
df_train = None
if os.path.exists(DATASET_PATH):
    df_train = pd.read_csv(DATASET_PATH)
    df_train.columns = df_train.columns.str.strip()

# ___________Predicting function
# here we pass the r2 value to this function and it returns predicted output
def predict_ports(r2_val):
    """Run model prediction and display results."""
    user_X     = np.array([[r2_val]])
    prediction = model.predict(user_X)[0]

    print("\n" + "=" * 50)
    print("         PORT PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Input r2 : {r2_val}")
    print("-" * 50)
    print(f"  {'Port':<10} {'Predicted Value':>20}")
    print("-" * 50)
    for port_name, val in zip(PORTS, prediction):
        print(f"  {port_name:<10} {val:>20.6f}")
    print("=" * 50)
    return prediction

# ________Ploting the predict result
def plot_prediction(r2_val, prediction):
    """Plot predicted values against training data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"User Prediction  |  r2={r2_val}", fontsize=14, fontweight='bold')

    for ax, port, pred_val in zip(axes.flatten(), PORTS, prediction):
        if df_train is not None:
            # Sort by r2 for a clean line plot
            df_sorted = df_train.sort_values('r2')
            ax.plot(df_sorted['r2'].values, df_sorted[port].values,
                    color='steelblue', alpha=0.6, linewidth=1.5,
                    label='Training data')

            # Green vertical line at the input r2
            ax.axvline(r2_val, color='green', linewidth=1.5,
                       linestyle=':', label=f'r2 = {r2_val}')

        # Red horizontal line showing predicted value level
        ax.axhline(pred_val, color='red', linewidth=1.5,
                   linestyle='--', alpha=0.5, label=f'Predicted: {pred_val:.4f}')

        # ✅ Red dot exactly at the predicted point (intersection)
        ax.plot(r2_val, pred_val,
                'o',                      # circle marker
                color='red',
                markersize=10,
                zorder=5,                 # draw on top of everything
                label=f'▶ Point: ({r2_val}, {pred_val:.4f})')

        # Annotate the dot with its value
        ax.annotate(f'  {pred_val:.4f}',
                    xy=(r2_val, pred_val),
                    fontsize=9,
                    color='red',
                    fontweight='bold',
                    va='bottom')

        ax.set_title(port)
        ax.set_xlabel("r2")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/user_prediction.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"   Plot saved to '{save_path}'")


def predict_from_csv(csv_path):
    """Batch predict from a CSV file containing an r2 column."""
    if not os.path.exists(csv_path):
        print(f"   File not found: {csv_path}")
        return

    batch_df = pd.read_csv(csv_path)
    batch_df.columns = batch_df.columns.str.strip()

    if 'r2' not in batch_df.columns:
        print("   CSV must contain an 'r2' column.")
        return

    out_of_range = batch_df[(batch_df['r2'] < 0.10) | (batch_df['r2'] > 0.29)]
    if len(out_of_range) > 0:
        print(f"\n   Warning: {len(out_of_range)} rows have r2 outside"
              f" trained range [0.10 – 0.29]. Predictions may be inaccurate.")

    X_batch = batch_df[['r2']].values
    preds   = model.predict(X_batch)

    pred_df   = pd.DataFrame(preds, columns=[f'{p}_pred' for p in PORTS])
    output_df = pd.concat([batch_df[['r2']], pred_df], axis=1)

    save_path = "model/batch_predictions.csv"
    output_df.to_csv(save_path, index=False)

    print(f"\n   Batch predictions saved to '{save_path}'")
    print(output_df.to_string(index=False))


print("\n  Options:")
print("    [1] Single prediction  (enter r2 manually)")
print("    [2] Batch prediction   (load a CSV file)")
print("    [3] Exit")

while True:
    print()
    choice = input("Select option (1 / 2 / 3): ").strip()
    if choice == '1':
        while True:
            r2_input = input("\n  Enter r2 value (or 'back' to return): ").strip()
            if r2_input.lower() == 'back':
                break
            try:
                r2_val = float(r2_input)
            except ValueError:
                print("  Invalid value. Please enter a number.")
                continue
            if not (0.10 <= r2_val <= 0.29):
                print(f"\n  Warning: r2={r2_val} is outside the trained range"
                      f" [0.10 – 0.29]. Prediction may be inaccurate.")
            prediction = predict_ports(r2_val)
            show_plot = input("\n  Show prediction plot? (y/n): ").strip().lower()
            if show_plot == 'y':
                plot_prediction(r2_val, prediction)
            again = input("\n  Predict again? (y/n): ").strip().lower()
            if again != 'y':
                break
    elif choice == '2':
        csv_path = input("\n  Enter path to CSV file: ").strip()
        predict_from_csv(csv_path)
    elif choice == '3':
        print("\n  Goodbye!")
        break
    else:
        print("  Invalid choice. Please enter 1, 2, or 3.")