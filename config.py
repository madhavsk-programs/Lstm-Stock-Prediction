# config.py

DATA_PATH = "data/AAPL.csv"

# Time series parameters
WINDOW_SIZE = 30 
TRAIN_SPLIT = 0.8  

# Model parameters
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
EPOCHS = 40
LEARNING_RATE = 0.001
BATCH_SIZE = 32