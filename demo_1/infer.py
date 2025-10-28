import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np

# from google.colab import drive

# drive.mount("/content/drive")


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


# load model
gcms_model_path = "gcms_encoder_2025-10-18 17:46:47.181752.pt"
sensor_model_path = "sensor_encoder_2025-10-18 17:46:47.181755.pt"

hidden_dim = 128
embedding_dim = 16

gcms_encoder = GCMSDataEncoder(10, hidden_dim, embedding_dim)
sensor_encoder = SensorDataEncoder(13, hidden_dim, embedding_dim)

gcms_encoder.load_state_dict(torch.load(gcms_model_path))
sensor_encoder.load_state_dict(torch.load(sensor_model_path))

# uploading gcms data
df = pd.read_csv("gcms_dataframe.csv")

# adding ambient to the df
ambient_row = pd.DataFrame(
    [
        {
            "food_name": "ambient",
            "C": 0,
            "Ca": 0,
            "H": 0,
            "K": 0,
            "Mg": 0,
            "N": 0,
            "Na": 0,
            "O": 0,
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

import torch.nn.functional as F


def evaluate_retrieval(
    smell_matrix, gcms_data, gcms_encoder, sensor_encoder, device="cpu"
):
    gcms_encoder.eval()
    sensor_encoder.eval()

    smell_matrix = torch.tensor(smell_matrix, dtype=torch.float)
    gcms_data = torch.tensor(gcms_data, dtype=torch.float)

    with torch.no_grad():
        gcms_data = gcms_data.to(device)
        smell_matrix = smell_matrix.to(device)

        z_gcms = gcms_encoder(gcms_data)
        z_sensor = sensor_encoder(smell_matrix)

        z_gcms = F.normalize(z_gcms, dim=1)
        z_sensor = F.normalize(z_sensor, dim=1)

        sim = torch.matmul(z_sensor, z_gcms.t())
        predicted = sim.argmax(dim=1)

    return predicted.tolist()


# Example: placeholder for actual smell data
smell_matrix = np.zeros(
    (1, 13), dtype=float
)  # TODO: replace with real collected smell matrix

predicted_from_smell = evaluate_retrieval(
    smell_matrix, gcms_data, gcms_encoder, sensor_encoder
)

print([ix_to_name[ix] for ix in predicted_from_smell])