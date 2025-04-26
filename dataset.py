import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import json


def get_datasets(config):
    df = pd.read_pickle(config["data_dir"])

    prc = df[(df["precipitation"] > 0.5) & (df["precipitation"] < 3)]
    prcWH = prc.copy()

    prcWH["pr_lag1"] = prcWH["precipitation"].shift(1).fillna(0)
    prcWH["prc_roll3"] = prcWH["precipitation"].rolling(window=3, min_periods=1).mean()
    prcWH["bin"] = prcWH["precipitation"].apply(lambda x: 1 if x > 0.2 else 0)
    prcWH["ps_diff_12"] = prcWH["ps1"] - prcWH["ps2"]
    prcWH["ps_diff_13"] = prcWH["ps1"] - prcWH["ps3"]
    prcWH["net_rad1"] = df["rsds1"] - prcWH["rsus1"]
    prcWH["net_rad2"] = df["rsds2"] - prcWH["rsus2"]
    prcWH["clt_diff_12"] = prcWH["clt1"] - prcWH["clt2"]
    prcWH["clt_diff_13"] = prcWH["clt1"] - prcWH["clt3"]

    prcWH["precipitation"] *= 25.4
    prcWH["month1"] = np.sin(2 * np.pi * prcWH["Datetime"].dt.month / 12)
    prcWH["month"] = prcWH["Datetime"].dt.month

    X = (
        prcWH.drop(
            [
                "City",
                "lat1",
                "lon1",
                "lat2",
                "lon2",
                "lat3",
                "lon3",
                "precip_binary",
            ],
            axis=1,
        )
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    dateTimes = X.pop("Datetime")
    y = X["precipitation"]
    X = X.drop("precipitation", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = (
        X_scaled[: -int(len(X_scaled) * 0.2)],
        X_scaled[-int(len(X_scaled) * 0.2) :],
    )
    y_train, y_test = y.iloc[: -int(len(y) * 0.2)], y.iloc[-int(len(y) * 0.2) :]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])

    return train_loader, test_loader, X_train.shape[1]
