import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class GCMSDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SensorDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# from google.colab import drive

# drive.mount("/content/drive")

# uploading gcms data
df = pd.read_csv("gcms_dataframe.csv")

# adding ambient to the df
ambient_row = pd.DataFrame(
    [
        {
            "food_name": "ambient",
            "C": 0.04,
            "Ca": 0,
            "H": 0.00005,
            "K": 0,
            "Mg": 0,
            "N": 78.08,
            "Na": 0,
            "O": 20.95,
            "P": 0,
            "Se": 0,
        }
    ]
)

df = pd.concat([df, ambient_row], ignore_index=True)

# getting rid of names and keeping only numerical values
df_dropped = df.drop(columns=["food_name"], errors="ignore")
gcms_data = df_dropped.values

available_food_names = df["food_name"].to_list()
ix_to_name = {i: name for i, name in enumerate(available_food_names)}
name_to_ix = {name: i for i, name in enumerate(available_food_names)}

available_food_names

# loading smell sensor data
smell_data_path = "smell_data_switch"

paths = []
for file in os.listdir(smell_data_path):
    file_path = os.path.join(smell_data_path, file)
    food_name = file.split(".")[0]
    if food_name in available_food_names:
        paths.append(file_path)


def create_state_average_df(df):
    df["Group"] = (df["State"] != df["State"].shift()).cumsum()
    averaged_df = df.groupby("Group").mean().reset_index()
    averaged_df["State"] = df.groupby("Group")["State"].first().values
    averaged_df = averaged_df.drop(columns=["Group"])
    averaged_df = averaged_df[averaged_df["State"] < 2]
    averaged_df.reset_index(drop=True)
    return averaged_df


def calculate_state_difference(df):
    if df.iloc[0]["State"] != 1:
        df = df.iloc[1:].reset_index(drop=True)
    if len(df) % 2 != 0:
        df = df[:-1]
    odd_rows = df.iloc[1::2].reset_index(drop=True)
    even_rows = df.iloc[0::2].reset_index(drop=True)
    result = odd_rows - even_rows
    return result


from collections import defaultdict
import re

ingredient_df = []

for path in paths:
    ingredient_name = re.split(r"[./]", path)[-3]
    dataframe = pd.read_csv(path)
    if dataframe.shape[1] > 14:
        dataframe = dataframe[dataframe.columns[:14]]
    dataframe.drop(columns="timestamp", inplace=True)
    dataframe.rename(columns={dataframe.columns[-1]: "State"}, inplace=True)
    avg_ingredient_df = create_state_average_df(dataframe)
    diff_ingredient_df = calculate_state_difference(avg_ingredient_df)
    diff_ingredient_df["label"] = name_to_ix[ingredient_name]
    ingredient_df.append(diff_ingredient_df)

combined_df = pd.concat(ingredient_df, axis=0, ignore_index=True)
columns_to_normalize = combined_df.columns[:13]


def filter_outliers(group):
    numerical_columns = group.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        Q1 = group[col].quantile(0.25)
        Q3 = group[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        group = group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
    return group


filtered_groups = (
    combined_df.groupby("label").apply(filter_outliers).reset_index(drop=True)
)
print("Filtered DataFrame:")
print(filtered_groups)

scaler = StandardScaler()
numerical_columns = filtered_groups.select_dtypes(include=[np.number]).columns
numerical_columns = numerical_columns.drop(["label", "State"])
filtered_groups[numerical_columns] = scaler.fit_transform(
    filtered_groups[numerical_columns]
)
print("\nNormalized DataFrame:")
print(filtered_groups)


def select_median_representative(group, n=1):
    median_values = group.median()
    distances = np.linalg.norm(group - median_values, axis=1)
    print(distances)
    group["distance"] = distances
    closest_rows = group.nsmallest(n, "distance").drop(columns="distance")
    return closest_rows


def select_median(group):
    median_values = group.median()
    return median_values


label_counts = filtered_groups.groupby("label").size().reset_index(name="count")
label_counts

sampled_df = filtered_groups.groupby("label").apply(select_median)

df_tuples = filtered_groups.apply(tuple, axis=1)
representatives_tuples = sampled_df.apply(tuple, axis=1)
remaining_data = filtered_groups[~df_tuples.isin(representatives_tuples)]

remaining_data.shape

smell_data = sampled_df.drop(["label", "State"], axis=1).values
y = sampled_df["label"].values

gcms_data.shape

pair_data = []
for i in range(len(smell_data)):
    pair_data.append((smell_data[i], gcms_data[int(y[i])]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


def cross_modal_contrastive_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    sim = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    loss = 0.5 * (loss_12 + loss_21)
    return loss


class PairedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gcms_vec, smell_vec = self.data[idx]
        gcms_vec = torch.tensor(gcms_vec, dtype=torch.float)
        smell_vec = torch.tensor(smell_vec, dtype=torch.float)
        return gcms_vec, smell_vec


dataset = PairedDataset(pair_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(gcms_data.shape)
print(smell_data.shape)

gcms_input_dim = gcms_data.shape[1]
sensor_input_dim = smell_data.shape[1]
embedding_dim = 16
hidden_dim = 128
temperature = 0.07
num_epochs = 100

gcms_encoder = GCMSDataEncoder(gcms_input_dim, hidden_dim, embedding_dim)
sensor_encoder = SensorDataEncoder(sensor_input_dim, hidden_dim, embedding_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gcms_encoder.to(device)
sensor_encoder.to(device)
params = list(gcms_encoder.parameters()) + list(sensor_encoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)

for epoch in range(num_epochs):
    gcms_encoder.train()
    sensor_encoder.train()
    total_loss = 0.0
    for x_sensor, x_gcms in dataloader:
        x_gcms = x_gcms.to(device)
        x_sensor = x_sensor.to(device)
        optimizer.zero_grad()
        z_gcms = gcms_encoder(x_gcms)
        z_sensor = sensor_encoder(x_sensor)
        loss = cross_modal_contrastive_loss(z_gcms, z_sensor, temperature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_retrieval(
    test_smell_data, test_smell_label, gcms_encoder, sensor_encoder, device="cpu"
):
    gcms_encoder.eval()
    sensor_encoder.eval()
    all_z_gcms = []
    all_z_sensor = []
    testing_gcms_data = torch.tensor(gcms_data, dtype=torch.float).to(device)
    gcms_embeddings = gcms_encoder(testing_gcms_data)
    z_gcms = F.normalize(gcms_embeddings, dim=1)
    test_smell_data = torch.tensor(test_smell_data, dtype=torch.float).to(device)
    smell_embeddings = sensor_encoder(test_smell_data)
    z_smell = F.normalize(smell_embeddings, dim=1)
    sim = torch.matmul(z_smell, z_gcms.T)
    print(f"Similarity matrix shape: {sim.shape}")
    predicted = sim.argmax(dim=1)
    print("------------------Predictions---------------------")
    print(predicted)
    correct = predicted == test_smell_label
    accuracy = correct.float().mean().item()
    precision = precision_score(test_smell_label, predicted, average="macro")
    recall = recall_score(test_smell_label, predicted, average="macro")
    f1 = f1_score(test_smell_label, predicted, average="macro")
    conf_matrix = confusion_matrix(test_smell_label, predicted)
    print("------------------Test Statistics---------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return accuracy, conf_matrix


test_smell_data = remaining_data.drop(["label", "State"], axis=1).values
test_y = remaining_data["label"].values
print(test_smell_data.shape)

accuracy, conf_matrix = evaluate_retrieval(
    test_smell_data, test_y, gcms_encoder, sensor_encoder, device=device
)
print(f"Test retrieval accuracy: {accuracy*100:.2f}%")

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def analyze_confusion_matrix(conf_matrix):
    num_classes = conf_matrix.shape[0]
    class_metrics = {}
    for i in range(num_classes):
        num_predictions = np.sum(conf_matrix[i])
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / num_predictions
        class_metrics[available_food_names[i]] = {"Accuracy": precision}
    return class_metrics


analyze_confusion_matrix(conf_matrix)

from datetime import datetime

print(datetime.now())

gcms_model_path = f"gcms_encoder_{datetime.now()}.pt"
sensor_model_path = f"sensor_encoder_{datetime.now()}.pt"
torch.save(gcms_encoder.state_dict(), gcms_model_path)
torch.save(sensor_encoder.state_dict(), sensor_model_path)
