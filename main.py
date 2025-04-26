import os
import torch
import json

from src.models.transformer import PrecipTransformer
from src.training.train import train_model
from src.training.losses import custom_loss_fn
from src.data.dataset import get_datasets
from src.evaluation.evaluate import evaluate_predictions, predict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("src/config/config.json", "r") as f:
        config = json.load(f)

    os.makedirs(config["output_dir"], exist_ok=True)

    train_loader, test_loader, input_dim = get_datasets(config)

    model = PrecipTransformer(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = custom_loss_fn

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
        y_true, y_pred, title="Downformer Performance", output_dir=config["output_dir"]
    )


if __name__ == "__main__":
    main()
