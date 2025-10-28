import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

from data_loader import load_sensor_data, build_sliding_data, diff_data_like
from models.transformer import Transformer
from config import CLASS_LABELS, DEVICE, DATA_PATH, SAVE_DIR, EPOCHS, LEARNING_RATE


<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
WINDOW_SIZE = 60
STRIDE = 5
CHANNELS = [0, 1, 2, 3]
PROCESSING = ["standard_scaler", "diff_data_like"]
CONFIG_NAME = "long_overlap_first_4"

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
os.makedirs(SAVE_DIR, exist_ok=True)


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return 100 * correct / max(total, 1)


def train():
    print(f"🚀 Training {CONFIG_NAME} ({PROCESSING})")

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    train_data, _ = load_sensor_data(DATA_PATH)
    if "diff_data_like" in PROCESSING:
        train_data = diff_data_like(train_data)

    X_train, y_train = build_sliding_data(train_data, WINDOW_SIZE, STRIDE)
    X_train = X_train[:, :, CHANNELS]

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape
    )

<<<<<<< HEAD
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

=======
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    val_ratio = 0.2
    n_val = int(len(X_train) * val_ratio)
    n_train = len(X_train) - n_val
    train_ds, val_ds = random_split(
        TensorDataset(X_train, y_train),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    num_classes = len(CLASS_LABELS)
    input_dim = len(CHANNELS)
    model = Transformer(
        input_dim=input_dim,
        model_dim=128,
        num_classes=num_classes,
        num_heads=4,
        num_layers=3,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience, no_improve = 5, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

        train_acc = 100 * total_correct / total_samples
        val_acc = evaluate(model, val_loader, DEVICE, num_classes)
        print(
            f"Epoch {epoch:02d} | Loss={total_loss/total_samples:.4f} | Train={train_acc:.2f}% | Val={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping.")
                break

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        SAVE_DIR,
        f"{CONFIG_NAME}_win{WINDOW_SIZE}_str{STRIDE}_proc{'_'.join(PROCESSING)}_{timestamp}.pt",
    )
    scaler_path = os.path.join(SAVE_DIR, f"{CONFIG_NAME}_scaler_{timestamp}.pkl")

    torch.save(best_state or model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Model saved: {model_path}")
    print(f"💾 Scaler saved: {scaler_path}")
    print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()