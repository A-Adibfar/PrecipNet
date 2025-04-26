import torch
from tqdm import tqdm


def train_model(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epochs=100,
    patience=10,
    output_path="best_model.pth",
):
    model.train()
    model.to(device)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            print("✅ Validation loss improved — model saved.")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break
