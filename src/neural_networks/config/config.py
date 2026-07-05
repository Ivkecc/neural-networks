# Dataset
DATASET_ROOT = "data/raw/UCI-HAR Dataset"

INPUT_LENGTH = 128
INPUT_CHANNELS = 6
NUM_CLASSES = 6

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50

# Reproducibility
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

MODEL_TYPE = "cnn_lstm"