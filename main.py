import torch
from torch.utils.data import DataLoader
from model import PrecipTransformer
from train import train_model
from evaluate import predict, evaluate_predictions
from dataset import get_datasets
import json
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open("config.json"))
    os.makedirs(config["output_dir"], exist_ok=True)

    train_loader, test_loader, input_dim = get_datasets(config)

    model = PrecipTransformer(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    loss_fn = lambda pred, target: (
        0.6
        * (
            1
            - (
                1
                - torch.sum((target - pred) ** 2)
                / (torch.sum((target - torch.mean(target)) ** 2) + 1e-8)
            )
        )
        + 0.4 * torch.mean((pred - target) ** 2)
    )

    train_model(
        model,
        train_loader,
        optimizer,
        loss_fn,
        device,
        epochs=config["epochs"],
        patience=config["patience"],
        output_path=os.path.join(config["output_dir"], "best_model.pth"),
    )

    model.load_state_dict(
        torch.load(os.path.join(config["output_dir"], "best_model.pth"))
    )
    y_true, y_pred = predict(model.to("cpu"), test_loader)
    evaluate_predictions(
        y_true,
        y_pred,
        title="Transformer Model Performance",
        output_dir=config["output_dir"],
    )


if __name__ == "__main__":
    main()
