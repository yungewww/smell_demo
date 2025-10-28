import torch





WINDOW_CONFIGS = {
    "long_overlap": {"window_size": 60, "stride": 5},  
    
    
    
    
}

CHANNEL = {
    0: "NO2",
    1: "C2H5OH",
    2: "VOC",
    3: "CO",
    4: "Alcohol",
    5: "LPG",
    6: "Benzene",
    7: "Temperature",
    8: "Pressure",
    9: "Humidity",
    10: "Gas_Resistance",
    11: "Altitude",
}

SELECTED_INDICES = {

    
    
    "first_4": [0, 1, 2, 3],
    "no_benzene": [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,

    ],  
    "no_benzene_no_alcohol": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11],
}

DATA_PROCESSING = ["standard_scaler", "diff_data_like", "highpass_fft_batch"]

DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 30

CLASS_LABELS = {
    "cloves": 0,
    "cumin": 1,
    "oregano": 2,
}



SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2
SAVE_DIR = "checkpoints"
DATA_PATH = "data"



if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # ✅ Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # ✅ NVIDIA GPU
else:
    DEVICE = torch.device("cpu")  # ✅ CPU fallback
print(f"🖥 Using device from config: {DEVICE}")