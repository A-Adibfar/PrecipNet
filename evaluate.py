import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


def predict(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).squeeze()
            all_preds.append(preds.numpy())
            all_labels.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_true, y_pred


def evaluate_predictions(y_true, y_pred, title="Model Evaluation", output_dir="."):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"🔹 {title}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")

    results_df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred})
    results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "residuals_hist.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True Precipitation")
    plt.ylabel("Predicted Precipitation")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
    plt.close()
