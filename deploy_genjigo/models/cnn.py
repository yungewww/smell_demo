import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm




data_path = "task4/data"  
seconds = 60  
sampling_rate = 1  
channels = 12
batch_size = 16
epochs = 50
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class SensorDataset(Dataset):
    def __init__(self, base_dir):
        self.samples = []
        self.labels = []
        self.class_names = sorted(os.listdir(base_dir))
        self.max_len = seconds * sampling_rate

        for label_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for f in os.listdir(class_dir):
                if not f.endswith(".csv"):
                    continue
                path = os.path.join(class_dir, f)
                df = pd.read_csv(path)

                df = df.iloc[: self.max_len, :channels]  
                if len(df) < self.max_len:
                    
                    pad = np.zeros((self.max_len - len(df), df.shape[1]))
                    df = np.vstack([df.values, pad])
                else:
                    df = df.values

                
                df = StandardScaler().fit_transform(df)
                self.samples.append(df)
                self.labels.append(label_idx)

        self.samples = np.stack(self.samples)
        self.labels = np.array(self.labels)
        print(f"✅ Loaded {len(self.samples)} samples from {base_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        # shape [C, T]
        x = torch.tensor(x, dtype=torch.float32).T.unsqueeze(
            0
        )  # [1, channels, timesteps]
        return x, torch.tensor(y, dtype=torch.long)




class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (channels // 4) * (seconds * sampling_rate // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)




train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")

train_dataset = SensorDataset(train_dir)
test_dataset = SensorDataset(test_dir)

num_classes = len(train_dataset.class_names)
print(f"Classes: {train_dataset.class_names}")



val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)




model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)




def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * total_correct / total




@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_correct, total = 0, 0
    per_class_total = torch.zeros(num_classes)
    per_class_correct = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
        for c in range(num_classes):
            mask = y == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += ((preds == y) & mask).sum().item()
    acc = 100.0 * total_correct / total
    per_class_acc = 100.0 * (per_class_correct / per_class_total.clamp_min(1))
    return acc, per_class_acc




best_val = 0
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_acc, _ = evaluate(model, val_loader)
    print(
        f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%"
    )
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "task4/checkpoints/cnn_best.pt")
        print(f"💾 Saved best model ({val_acc:.2f}%)")




# model.load_state_dict(torch.load("task4/checkpoints/cnn_best.pt", map_location=device))
# test_acc, per_class_acc = evaluate(model, test_loader)
# print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
# for cls, acc in zip(train_dataset.class_names, per_class_acc):
#     print(f"{cls:<15}: {acc:.2f}%")



model.load_state_dict(torch.load("task4/checkpoints/cnn_best.pt", map_location=device))


train_acc, per_class_acc_train = evaluate(model, train_loader)

test_acc, per_class_acc_test = evaluate(model, test_loader)

print("\n📊 Evaluation Results")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test  Accuracy: {test_acc:.2f}%")

print("\n📊 Per-Class Accuracy (Train)")
for cls, acc in zip(train_dataset.class_names, per_class_acc_train):
    print(f"{cls:<15}: {acc:.2f}%")

print("\n📊 Per-Class Accuracy (Test)")
for cls, acc in zip(train_dataset.class_names, per_class_acc_test):
    print(f"{cls:<15}: {acc:.2f}%")